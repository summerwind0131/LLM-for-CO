# =============================================================================
#  模块七：核心求解器
# =============================================================================

import random

import numpy as np

from .config import (
    USE_TWO_OPT,
    USE_ELITE_RESTART,
    LLM_TRIGGER_ONLY_ON_STAGNATION,
    DESTRUCTION_RATIO,
    EPSILON,
    MIN_PROB,
    ALNS_RHO,
    BASE_PROBS,
    DESTROY_OPS,
)
from .utils import calc_route_distance, calc_trigger_interval, select_by_roulette
from .operators import (
    OPERATORS,
    apply_two_opt,
    destroy_random,
    repair_greedy,
)
from .state import state_to_meta_json
from .llm_brain import ask_llm_for_mode


def run_solver(
    nodes:         np.ndarray,
    distances:     np.ndarray,
    initial_route: list,
    strategy:      str = "baseline",
    solver_seed:   int = 42,
) -> tuple:
    """
    统一求解器，支持三种策略：baseline / traditional_alns / sc_llm_os。

    返回: (distance_history, best_distance, llm_log, has_dirty_data)
    """
    # ── 种子固定 ─────────────────────────────────────────────────────────────
    random.seed(solver_seed)
    np.random.seed(solver_seed)

    # ── 初始解构造 [EXP-8] ────────────────────────────────────────────────────
    # 使用传入的、统一生成的初始解
    current_route = initial_route.copy()
    if USE_TWO_OPT:
        current_route = apply_two_opt(current_route, distances)

    best_route    = current_route.copy()
    best_distance = calc_route_distance(best_route, distances)

    # ── 状态变量 ─────────────────────────────────────────────────────────────
    stagnation_counter = 0
    distance_history   = [best_distance]
    llm_log            = []
    has_dirty_data     = False

    num_cities = len(nodes)
    NUM_ITERATIONS = 200 + num_cities * 5
    STAGNATION_RESET_THRESHOLD = max(30, int(num_cities * 0.3))

    llm_trigger_interval = calc_trigger_interval(num_cities, NUM_ITERATIONS)
    k_destroy = max(3, int(num_cities * DESTRUCTION_RATIO))

    # [EXP-7] 缓存当前解距离，避免主循环内重复O(N)计算
    current_dist = best_distance

    elite_route             = current_route.copy()
    elite_distance          = best_distance
    no_improve_since_elite  = 0
    ELITE_RESTART_THRESHOLD = max(80, len(nodes) // 2)
    EMERGENCY_STAGNATION    = 35
    last_trigger_iter       = 0

    # [EXP-9] Trad-ALNS冷启动平滑：Laplace伪计数，首轮avg_score≠0
    op_stats = {
        op: {"used": 1, "improved": 0, "score": 1.0}
        for op in DESTROY_OPS
    }
    current_weights  = BASE_PROBS.copy()
    phase_start_dist = best_distance

    print(f"\n🚀 [{strategy.upper():<18}] 启动 | "
          f"初始距离: {best_distance:.1f}")

    # ── 主迭代循环 ────────────────────────────────────────────────────────────
    for iteration in range(1, NUM_ITERATIONS + 1):

        # 计算当前进度百分比 (0.0 到 1.0)
        progress = iteration / NUM_ITERATIONS

        # 新降温公式：随着进度从 10% 初始温度平滑衰减到近乎 0
        # 使用指数衰减，保证在整个 NUM_ITERATIONS 周期内都有合理的温度分布
        sa_temperature = best_distance * 0.1 * ((0.01) ** progress)

        # ── 阶段性策略调控 ────────────────────────────────────────────────────
        emergency_trigger = (
            strategy == "sc_llm_os"
            and stagnation_counter >= EMERGENCY_STAGNATION
            and (iteration - last_trigger_iter) >= 30
        )

        # 触发检测：如果设置了"仅在停滞(应急)时触发 LLM"，则忽略固定周期的那部分判断（对于 sc_llm_os 而言），仅依靠 emergency_trigger。
        # 考虑到 baseline 和 traditional_alns 依然需要按固定周期演算统计，做区分控制。
        is_trigger_time = False
        if strategy == "sc_llm_os" and LLM_TRIGGER_ONLY_ON_STAGNATION:
            is_trigger_time = emergency_trigger
        else:
            is_trigger_time = (iteration % llm_trigger_interval == 0) or emergency_trigger

        if is_trigger_time:
            if emergency_trigger and iteration % llm_trigger_interval != 0:
                print(f"   [⚡ 应急触发] iter={iteration} "
                      f"stagnation={stagnation_counter}")
            last_trigger_iter = iteration

            phase_distance_drop = phase_start_dist - best_distance

            # ── Baseline：均匀随机权重 ────────────────────────────────────────
            if strategy == "baseline":
                current_weights = {op: 1.0 / len(DESTROY_OPS)
                                   for op in DESTROY_OPS}

            # ── Traditional ALNS：指数平滑自适应权重 ─────────────────────────
            elif strategy == "traditional_alns":
                for op in DESTROY_OPS:
                    used      = op_stats[op]["used"]
                    avg_score = op_stats[op]["score"] / used  # 有伪计数，分母≥1
                    current_weights[op] = (
                        (1 - ALNS_RHO) * current_weights[op]
                        + ALNS_RHO * avg_score
                    )
                # 归一化 + MIN_PROB 保底
                total_w = sum(current_weights.values()) or 1.0
                current_weights = {
                    op: max(w / total_w, MIN_PROB)
                    for op, w in current_weights.items()
                }
                total_w2 = sum(current_weights.values())
                current_weights = {
                    op: w / total_w2
                    for op, w in current_weights.items()
                }

            # ── SC-LLM-OS：LLM宏观指令 + 偏置调权 ───────────────────────────
            elif strategy == "sc_llm_os":
                meta_json = state_to_meta_json(
                    current_iter        = iteration,
                    num_iterations      = NUM_ITERATIONS,
                    stagnation          = stagnation_counter,
                    route               = current_route,
                    distances           = distances,
                    op_stats            = op_stats,
                    phase_distance_drop = phase_distance_drop,
                    best_distance       = best_distance,
                )
                decision = ask_llm_for_mode(meta_json)
                mode     = decision.get("mode", "explore_global")

                # [修复A] phase_drop=0 且已连续停滞：强制禁止 exploit
                # 理由：整阶段零改善说明当前精细搜索方向已完全失效
                #       无论算子分数多高，继续exploit只是在原地打转
                if phase_distance_drop == 0 and stagnation_counter > 10:
                    if mode == "exploit":
                        mode = "explore_topology"
                        print(f"   [🛡️  零改善覆盖] exploit → explore_topology "
                              f"(phase_drop=0, stagnation={stagnation_counter})")
                    decision["reasoning"] += " [覆盖：整阶段零改善，禁止exploit]"

                if decision.get("is_fallback", False):
                    has_dirty_data = True
                    print(f"   ⚠️  [iter={iteration}] Fallback，本seed已标记脏数据")

                print(f"   [🧠 iter={iteration:>4d}] "
                      f"{decision.get('reasoning', '')[:60]} "
                      f"→ {mode}"
                      + (" [FB]" if decision.get("is_fallback") else ""))

                # 偏置：始终从全局固定BASE_PROBS出发（消除累积漂移）
                bias_map = {
                    "exploit":          np.array([1.5, 0.8, 0.8]),
                    "explore_topology": np.array([0.8, 1.5, 0.8]),
                    "explore_global":   np.array([0.8, 0.8, 1.5]),
                }
                p    = np.array([BASE_PROBS[op] for op in DESTROY_OPS])
                p    = p * bias_map.get(mode, np.ones(3))
                p    = p / p.sum()                       # 第一次归一化

                # tau温度缩放：停滞越深分布越平坦（更随机探索）
                tau  = 1.5 if stagnation_counter > 15 else 0.7
                p    = p ** (1.0 / tau)
                p    = p / p.sum()                       # 第二次归一化

                # MIN_PROB 探索保底
                p    = np.maximum(p, MIN_PROB)
                p    = p / p.sum()                       # 第三次归一化

                current_weights = {
                    op: float(v) for op, v in zip(DESTROY_OPS, p)
                }

                print(f"   [⚖️  weights] "
                      f"{ {k: f'{v:.3f}' for k, v in current_weights.items()} }")

                llm_log.append({
                    "iteration":    iteration,
                    "stagnation":   stagnation_counter,
                    "mode":         mode,
                    "reasoning":    decision.get("reasoning", ""),
                    "is_fallback":  decision.get("is_fallback", False),
                    "weights":      {k: round(v, 4)
                                     for k, v in current_weights.items()},
                    "best_distance": round(best_distance, 2),
                    "phase_drop":   round(phase_distance_drop, 2),
                })

            # 重置阶段统计（保留伪计数基底）
            op_stats = {
                op: {"used": 1, "improved": 0, "score": 1.0}
                for op in DESTROY_OPS
            }
            phase_start_dist = best_distance

        # ── 停滞计数器独立重置 ────────────────────────────────────────────────
        if stagnation_counter >= STAGNATION_RESET_THRESHOLD:
            stagnation_counter = 0

        # ── 算子执行：破坏 → 修复 → 局部搜索 ────────────────────────────────
        chosen           = select_by_roulette(current_weights)
        partial, removed = OPERATORS["destroy"][chosen](
            current_route, k_destroy, distances
        )
        new_route        = OPERATORS["repair"]["greedy"](
            partial, removed, distances
        )
        if USE_TWO_OPT:
            new_route = apply_two_opt(new_route, distances)
        new_distance = calc_route_distance(new_route, distances)

        # ── 算子评分（相对改善幅度连续奖励）─────────────────────────────────
        delta = new_distance - current_dist
        op_stats[chosen]["used"] += 1

        if new_distance < best_distance:
            rel_imp = (best_distance - new_distance) / (best_distance + EPSILON)
            op_stats[chosen]["score"]    += 3.0 + rel_imp * 10.0
            op_stats[chosen]["improved"] += 1
            best_distance      = new_distance
            best_route         = new_route.copy()
            current_route      = new_route
            current_dist       = new_distance   # [EXP-7] 同步缓存
            stagnation_counter = 0

        elif delta < 0:
            # [EXP-4] current改善但全局最优未更新
            # 部分衰减停滞计数器，避免"幽灵停滞"
            stagnation_counter = max(0, stagnation_counter - 2)
            op_stats[chosen]["score"]    += 2.0
            op_stats[chosen]["improved"] += 1
            current_route = new_route
            current_dist  = new_distance    # [EXP-7] 同步缓存

        else:
            # SA 概率接受劣解
            accept_prob = np.exp(-delta / max(sa_temperature, 1e-5))
            if random.random() < accept_prob:
                current_route = new_route
                current_dist  = new_distance  # [EXP-7] 同步缓存
            stagnation_counter += 1           # [EXP-4] 只有未改善才累计

        distance_history.append(best_distance)

        # 更新精英存档
        if best_distance < elite_distance:
            elite_route            = best_route.copy()
            elite_distance         = best_distance
            no_improve_since_elite = 0
        else:
            no_improve_since_elite += 1

        # 精英解重启：长期无改善时主动逃脱
        if (USE_ELITE_RESTART
                and strategy == "sc_llm_os"
                and no_improve_since_elite >= ELITE_RESTART_THRESHOLD):

            # 破坏规模随重启次数递增（越来越激进）
            restart_k = min(
                k_destroy + 5,   # 比早期破坏规模更大
                len(current_route) // 4        # 不超过路径的1/4
            )
            r_partial, r_removed = destroy_random(
                elite_route, restart_k, distances
            )
            r_route   = repair_greedy(r_partial, r_removed, distances)
            if USE_TWO_OPT:
                r_route = apply_two_opt(r_route, distances)
            r_dist    = calc_route_distance(r_route, distances)

            # 接受条件：不比精英解差超过2%
            if r_dist <= elite_distance * 1.02:
                current_route           = r_route
                current_dist            = r_dist
                stagnation_counter      = 0
                no_improve_since_elite  = 0
                print(f"   [🔄 精英重启] iter={iteration} "
                      f"破坏{restart_k}节点 "
                      f"新起点:{r_dist:.0f} (精英:{elite_distance:.0f})")
            else:
                # 重启解太差，直接回到精英解
                current_route           = elite_route.copy()
                current_dist            = elite_distance
                stagnation_counter      = 0
                no_improve_since_elite  = 0
                print(f"   [🔄 精英重置] iter={iteration} "
                      f"重启解过差({r_dist:.0f})，回退精英解({elite_distance:.0f})")

    print(f"✅ [{strategy.upper():<18}] 完成 | 最优距离: {best_distance:.1f}"
          + (" ⚠️ 含脏数据" if has_dirty_data else ""))
    return distance_history, best_distance, best_route, llm_log, has_dirty_data
