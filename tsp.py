# v6.0
#加入矩阵检验
# 动态调整触发与迭代次数
# 加入更大规模实验
# =============================================================================
#  SC-LLM-OS: Strategic Commander LLM for Optimization Scheduling
#  最终实验版 v5.0  |  2026-04-05
#
#  相比v4.0的改动：
#  [EXP-1] USE_TWO_OPT=True，开启局部搜索，提升信号质量
#  [EXP-2] LLM_TRIGGER_INTERVAL 20→50，每阶段样本量充足
#  [EXP-3] NUM_ITERATIONS 300→600，给LLM更多观察窗口
#  [EXP-4] 修复stagnation虚高：current改善时不再累计停滞
#  [EXP-5] 修复destroy_segment：重建列表法彻底消除索引漂移
#  [EXP-6] 修复并行结果顺序：字典存储按seed组装
#  [EXP-7] current_dist缓存：消除主循环内重复O(N)计算
#  [EXP-8] 贪心初始解：替代纯随机打乱，提升初始解质量
#  [EXP-9] Trad-ALNS冷启动平滑：Laplace伪计数避免首轮塌缩
#  [EXP-10] API Key改为环境变量读取，源文件不再硬编码
#  [EXP-11] 跨平台中文字体自适应
# =============================================================================

import numpy as np
import random
import json
import urllib.request
import re
import os
import sys
import time
import math
import platform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from google import genai
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# 新增：缓存存取函数
def load_cache(instance_name: str) -> dict:
    cache_path = os.path.join(SCRIPT_DIR, f".cache_tsp_{instance_name}_opt2_{USE_TWO_OPT}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(instance_name: str, cache_data: dict):
    cache_path = os.path.join(SCRIPT_DIR, f".cache_tsp_{instance_name}_opt2_{USE_TWO_OPT}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

# 新增：基于问题规模自动计算触发间隔
def calc_trigger_interval(num_cities: int, num_iterations: int) -> int:
    """
    动态触发间隔：保证每次触发时每种算子至少有 MIN_SAMPLES_PER_OP 次使用
    从而保证统计可靠性不随规模下降。

    公式：interval = MIN_SAMPLES_PER_OP × len(DESTROY_OPS) × scale_factor
    """
    MIN_SAMPLES_PER_OP = 15      # 每种算子最少样本量
    n_ops              = 3   # 3

    # 基础间隔：保证每种算子至少15次
    base_interval = MIN_SAMPLES_PER_OP * n_ops  # = 45

    # 规模系数：节点数越多，每次迭代改善越难，需要更多样本
    if num_cities <= 60:
        scale = 0.8    # 小规模适当缩短
    elif num_cities <= 120:
        scale = 1.0    # 标准
    elif num_cities <= 180:
        scale = 1.5    # 中等规模加长
    else:
        scale = 2.5    # 大规模大幅加长

    interval = int(base_interval * scale)
    # 确保不超过总迭代数的1/5（至少触发5次）
    interval = min(interval, num_iterations // 5)
    return interval
# =============================================================================
# 模块零：控制台编码修复
# =============================================================================

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# 模块一：全局配置
# =============================================================================

# 尝试自动从 .env 文件加载环境变量（适用于服务器运行）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── API ───────────────────────────────────────────────────────────────────────
# LLM 供应商选择 (可选: "gemini" 或 "deepseek")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek").lower()

# Gemini 配置
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# DeepSeek 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()


# ── 实验规模 ──────────────────────────────────────────────────────────────────
INSTANCES  = ["ali535","d657","pr1002"]   # 多个规模用于泛化验证
SEEDS      = list(range(42, 72))  # [EXP-13] 30 个连续的整数，验证统计显著性
STRATEGIES = ["baseline", "traditional_alns", "sc_llm_os"]

# ── 算法超参数 ────────────────────────────────────────────────────────────────
# NUM_ITERATIONS 和 STAGNATION_RESET_THRESHOLD 将在各个实例加载后动态计算
# LLM_TRIGGER_INTERVAL 延后到加载数据时动态计算
DESTRUCTION_RATIO         = 0.06   # 动态破坏比例
USE_TWO_OPT               = False   # [EXP-1] 开启局部搜索
USE_ELITE_RESTART         = False  # [EXP-12] 精英解重启机制开关（当前停用）
LLM_TRIGGER_ONLY_ON_STAGNATION = True  # 仅在停滞(应急)时触发 LLM 策略调控，关闭固定周期触发
MAX_TWO_OPT_PASSES        = 2

# ── 数值常数 ──────────────────────────────────────────────────────────────────
EPSILON  = 1e-8
MIN_PROB = 0.1
ALNS_RHO = 0.5

# ── 固定基准概率（SC-LLM-OS偏置起点，消除累积漂移）─────────────────────────
BASE_PROBS  = {"worst": 0.4, "segment": 0.3, "random": 0.3}
DESTROY_OPS = ["worst", "segment", "random"]

# ── 并行配置 ──────────────────────────────────────────────────────────────────
NON_LLM_STRATEGIES   = ["baseline", "traditional_alns"]
MAX_PARALLEL_WORKERS = 4

# ── 输出目录 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ── 运行标识 ──────────────────────────────────────────────────────────────────
VERSION = "v7.0"
RUN_ID  = f"{VERSION}_{time.strftime('%Y%m%d_%H%M%S')}_{LLM_PROVIDER}"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sc_llm_os_results", RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 模块零点五：matplotlib 中文字体自适应 [EXP-11]
# =============================================================================

def _setup_matplotlib_fonts():
    sys_name = platform.system()
    if sys_name == "Windows":
        candidates = ["SimHei", "Microsoft YaHei", "FangSong", "DejaVu Sans"]
    elif sys_name == "Darwin":
        candidates = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    else:
        candidates = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]

    plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False

_setup_matplotlib_fonts()


# =============================================================================
# 模块二：数据加载
# =============================================================================

def load_tsplib_data(instance_name: str):
    """
    加载TSPLIB实例（本地不存在时自动下载）。
    距离公式：TSPLIB EUC_2D 标准 nint = floor(sqrt(...) + 0.5)
    使用NumPy向量化计算，兼顾标准精度与速度。
    """
    print(f"\n📦 正在加载数据集: {instance_name} ...")
    file_path = os.path.join(SCRIPT_DIR, "data", f"{instance_name}.tsp")

    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = (f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95"
               f"/tsp/{instance_name}.tsp.gz")
        print(f"   🌐 本地未找到，尝试下载: {url}")
        try:
            import gzip
            urllib.request.urlretrieve(url, file_path + ".gz")
            with gzip.open(file_path + ".gz", "rb") as f_in, \
                 open(file_path, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(file_path + ".gz")
            print("   ✅ 下载成功")
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            return None, None

    with open(file_path, "r") as f:
        content = f.read()

    nodes, parsing = [], False
    for line in content.split("\n"):
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            parsing = True
            continue
        if line in ("EOF", "") and parsing:
            parsing = False
            continue
        if parsing:
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                nodes.append((float(parts[1]), float(parts[2])))

    nodes = np.array(nodes)

    # TSPLIB EUC_2D 标准公式：nint(sqrt(dx²+dy²))
    diff      = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    distances = np.floor(
        np.sqrt((diff ** 2).sum(axis=-1)) + 0.5
    ).astype(np.int32)

    print(f"   ✅ 加载完成，共 {len(nodes)} 个城市。")
    return nodes, distances


# =============================================================================
# 模块三：工具函数
# =============================================================================

def fallback_calc_dist(route_arr: np.ndarray, distances: np.ndarray) -> int:
    n = len(route_arr)
    ans = 0
    for i in range(n):
        ans += distances[route_arr[i], route_arr[(i + 1) if (i + 1) < n else 0]]
    return ans

if HAS_NUMBA:
    numba_calc_dist = njit(fallback_calc_dist)

def calc_route_distance(route: list, distances: np.ndarray) -> int:
    route_arr = np.array(route, dtype=np.int32)
    if HAS_NUMBA:
        return int(numba_calc_dist(route_arr, distances))
    return int(fallback_calc_dist(route_arr, distances))


def select_by_roulette(weights: dict) -> str:
    total = sum(weights.values())
    if total <= 0:
        return random.choice(list(weights.keys()))
    pick, cumulative = random.uniform(0, total), 0.0
    for name, w in weights.items():
        cumulative += w
        if pick <= cumulative:
            return name
    return random.choice(list(weights.keys()))


# =============================================================================
# 模块四：算子库
# =============================================================================

def fallback_2opt(route_arr: np.ndarray, distances: np.ndarray, max_passes: int) -> np.ndarray:
    best = route_arr.copy()
    n    = len(best)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                i1 = i + 1
                j1 = (j + 1) % n
                delta = (  distances[best[i]] [best[j]]
                         + distances[best[i1]][best[j1]]
                         - distances[best[i]] [best[i1]]
                         - distances[best[j]] [best[j1]])
                if delta < -1e-8:
                    best[i1: j + 1] = best[i1: j + 1][::-1]
                    improved = True
        if not improved:
            break
    return best

if HAS_NUMBA:
    numba_2opt = njit(fallback_2opt)

def apply_two_opt(route: list, distances: np.ndarray) -> list:
    """
    增量 2-opt 局部搜索。
    针对大量节点优化，使用 numba 加速如果可用。
    """
    route_arr = np.array(route, dtype=np.int32)
    if HAS_NUMBA:
        best_arr = numba_2opt(route_arr, distances, MAX_TWO_OPT_PASSES)
    else:
        best_arr = fallback_2opt(route_arr, distances, MAX_TWO_OPT_PASSES)
    return best_arr.tolist()


def greedy_initial_route(num_cities: int, distances: np.ndarray,
                          start: int = None) -> list:
    """
    [EXP-8] 最近邻贪心构造初始解。
    质量远优于纯随机打乱（约为随机解的60%距离），
    给ALNS一个有意义的起点。
    """
    if start is None:
        start = random.randint(0, num_cities - 1)
    unvisited = list(range(num_cities))
    unvisited.remove(start)
    route = [start]
    while unvisited:
        curr    = route[-1]
        nearest = min(unvisited, key=lambda x: distances[curr][x])
        route.append(nearest)
        unvisited.remove(nearest)
    return route


# ── 破坏算子 ──────────────────────────────────────────────────────────────────

def destroy_random(route: list, k: int, distances=None):
    """随机移除 k 个节点"""
    rc      = route.copy()
    removed = [rc.pop(random.randint(0, len(rc) - 1)) for _ in range(k)]
    return rc, removed


def destroy_worst(route: list, k: int, distances: np.ndarray):
    """
    序贯式 Worst Destruction。
    每移除一个节点后重新计算所有剩余节点的移除代价，
    消除"独立性陷阱"（相邻节点代价互相影响）。
    """
    rc      = route.copy()
    removed = []
    for _ in range(k):
        n     = len(rc)
        costs = []
        for idx in range(n):
            prev = rc[idx - 1]
            curr = rc[idx]
            nxt  = rc[(idx + 1) % n]
            cost = (distances[prev][curr]
                    + distances[curr][nxt]
                    - distances[prev][nxt])
            costs.append((cost, idx))
        _, best_idx = max(costs, key=lambda x: x[0])
        removed.append(rc.pop(best_idx))
    return rc, removed


def destroy_segment(route: list, k: int, distances=None):
    """
    [EXP-5] 移除连续片段（重建列表法）。
    先按环形顺序记录待删节点值，再通过重建列表删除，
    彻底避免pop+索引漂移导致的非连续删除问题。
    支持路径中存在重复节点值的极端情况（计数消耗法）。
    """
    rc        = route.copy()
    n         = len(rc)
    start_idx = random.randint(0, n - 1)

    # 按环形顺序记录待删节点值
    removed = [rc[(start_idx + i) % n] for i in range(k)]

    # 计数消耗法重建列表（正确处理节点值重复的情况）
    counts = {}
    for node in removed:
        counts[node] = counts.get(node, 0) + 1

    new_rc = []
    for node in rc:
        if counts.get(node, 0) > 0:
            counts[node] -= 1      # 消耗一个删除配额
        else:
            new_rc.append(node)

    return new_rc, removed


# ── 修复算子 ──────────────────────────────────────────────────────────────────

def fallback_repair_greedy(route_arr: np.ndarray, removed_arr: np.ndarray, distances: np.ndarray) -> np.ndarray:
    rc = list(route_arr)
    for node in removed_arr:
        best_cost = 1e9
        best_idx  = 0
        n         = len(rc)
        for i in range(n):
            a    = rc[i]
            b    = rc[(i + 1) if (i + 1) < n else 0]
            cost = distances[a][node] + distances[node][b] - distances[a][b]
            if cost < best_cost:
                best_cost = cost
                best_idx  = i + 1
        rc.insert(best_idx if best_idx <= n else 0, node)
    return np.array(rc, dtype=np.int32)

if HAS_NUMBA:
    numba_repair_greedy = njit(fallback_repair_greedy)

def repair_greedy(route: list, removed: list, distances: np.ndarray) -> list:
    """
    贪心最小代价插入修复，使用 Numba JIT 优化以加速内循环 O(n) 操作。
    """
    if len(removed) == 0:
        return route.copy()
    route_arr = np.array(route, dtype=np.int32)
    removed_arr = np.array(removed, dtype=np.int32)
    if HAS_NUMBA:
        rc_arr = numba_repair_greedy(route_arr, removed_arr, distances)
    else:
        rc_arr = fallback_repair_greedy(route_arr, removed_arr, distances)
    return rc_arr.tolist()


# ── 算子注册表 ────────────────────────────────────────────────────────────────

OPERATORS = {
    "destroy": {
        "random":  destroy_random,
        "worst":   destroy_worst,
        "segment": destroy_segment,
    },
    "repair": {
        "greedy": repair_greedy,
    },
}


# =============================================================================
# 模块五：状态提取 S2N（State → Natural-language JSON）
# =============================================================================

def state_to_meta_json(
    current_iter:        int,
    num_iterations:      int,
    stagnation:          int,
    route:               list,
    distances:           np.ndarray,
    op_stats:            dict,
    phase_distance_drop: float,
    best_distance:       float,
) -> str:
    """
    将底层数学搜索状态映射为四维语义JSON，供LLM进行因果推理。
    四个维度：时序 / 拓扑 / 动力学 / 算子反馈
    """
    # 拓扑特征
    n_route   = len(route)
    edges     = [int(distances[route[i]][route[(i + 1) % n_route]])
                 for i in range(n_route)]
    avg_edge  = float(np.mean(edges))
    std_edge  = float(np.std(edges))
    long_ratio = sum(1 for e in edges if e > avg_edge + std_edge) / n_route

    # 算子表现整理
    fmt_ops = {}
    for op, stats in op_stats.items():
        used = stats["used"]
        fmt_ops[op] = {
            "used":         used,
            "success_rate": round(stats["improved"] / used, 3) if used > 0 else 0.0,
            "avg_score":    round(stats["score"]    / used, 3) if used > 0 else 0.0,
        }

    # 搜索阶段语义标签
    progress = current_iter / num_iterations
    phase    = ("early-exploration" if progress < 0.3 else
                "mid-exploitation"  if progress < 0.7 else
                "late-convergence")

    meta = {
        "temporal": {
            "progress": round(progress, 3),
            "phase":    phase,
        },
        "topological": {
            "avg_edge_length":          round(avg_edge, 2),
            "structural_anomaly_ratio": round(long_ratio, 3),
        },
        "dynamic": {
            "stagnation_steps":    stagnation,
            "is_deeply_stuck":     stagnation > 15,
            "phase_distance_drop": round(phase_distance_drop, 2),
            "current_best":        round(best_distance, 2),
        },
        "operator_feedback": fmt_ops,
    }
    return json.dumps(meta, indent=2, ensure_ascii=False)


# =============================================================================
# 模块六：LLM 大脑
# =============================================================================

def ask_llm_for_mode(meta_state_json: str) -> dict:
    """
    调用API 获取宏观战术模式决策。
    最多重试3次（指数退避），返回值含 is_fallback 字段标记数据质量。
    """
    prompt = f"""你是一个高级组合优化算法的宏观战略指挥官（Strategic Commander）。
你的任务是阅读当前搜索状态，基于算子历史表现进行因果归因，选择最合适的宏观战术模式。

【当前系统状态 JSON】
{meta_state_json}

【战术模式语义定义】
- exploit          (开发模式): 侧重精细微调（偏置worst算子）。
                               适用：搜索平稳下降，worst算子success_rate较高时。
- explore_topology (拓扑探索): 侧重切断路径中的交叉长边（偏置segment算子）。
                               适用：structural_anomaly_ratio较高，陷入局部最优时。
- explore_global   (全局探索): 引入彻底随机扰动（偏置random算子）。
                               适用：深度停滞(stagnation_steps>15)，所有算子均失效时。

【决策规则】
1. 优先参考 operator_feedback 中各算子的 avg_score，选择与高分算子匹配的模式。
2. 若所有算子分数均低于0.5且停滞严重，果断选择 explore_global。
3. 若处于 late-convergence 阶段，优先选择 exploit 保持精细收敛。

严格返回以下JSON，不输出任何其他内容：
{{
    "reasoning": "一句话：指出关键状态指标并说明选择该模式的因果逻辑",
    "mode": "<exploit|explore_topology|explore_global>"
}}
"""
    max_retry = 3

    for attempt in range(1, max_retry + 1):
        try:
            time.sleep(3)
            
            if LLM_PROVIDER == "gemini":
                client = genai.Client(api_key=API_KEY)
                res    = client.models.generate_content(
                    model    = "gemini-3.1-pro-preview",
                    contents = prompt,
                    config   = {
                        "response_mime_type": "application/json",
                        "temperature":        0.2,
                    },
                )
                result = json.loads(res.text)
                
            elif LLM_PROVIDER == "deepseek":
                url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
                }
                data = json.dumps({
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "reasoning_effort": "high",
                    "extra_body": {"thinking": {"type": "enabled"}},
                    "temperature": 0.2
                }).encode("utf-8")
                
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as response:
                    res_body = response.read().decode("utf-8")
                    content = json.loads(res_body)["choices"][0]["message"]["content"]
                    result = json.loads(content)
                    
            else:
                raise ValueError(f"未知的 LLM_PROVIDER: {LLM_PROVIDER}")

            if result.get("mode") not in (
                "exploit", "explore_topology", "explore_global"
            ):
                raise ValueError(f"非法mode字段: {result.get('mode')}")
            result["is_fallback"] = False
            return result

        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON解析失败 (attempt {attempt}): {e}")
        except Exception as e:
            print(f"   ⚠️  API调用失败 (attempt {attempt}): {e}")
            time.sleep(5 * attempt)

    print(f"   🔴 LLM连续失败{max_retry}次，启用Fallback。")
    return {
        "reasoning":   f"Fallback after {max_retry} failures",
        "mode":        "explore_global",
        "is_fallback": True,
    }


# =============================================================================
# 模块七：核心求解器
# =============================================================================

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
        
        # 触发检测：如果设置了“仅在停滞(应急)时触发 LLM”，则忽略固定周期的那部分判断（对于 sc_llm_os 而言），仅依靠 emergency_trigger。
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


# =============================================================================
# 模块八：并行加速（分层策略）
# =============================================================================

def _worker_non_llm(args: tuple) -> tuple:
    """
    进程池工作函数（仅供 baseline / traditional_alns 使用）。
    顶层函数定义保证 Windows 下 spawn 模式可正确序列化。
    """
    instance, nodes, distances, strategy, solver_seed, orig_seed, initial_route = args
    history, best_dist, best_route, _, _ = run_solver(
        nodes         = nodes,
        distances     = distances,
        initial_route = initial_route,
        strategy      = strategy,
        solver_seed   = solver_seed,
    )
    return strategy, orig_seed, history, best_dist, best_route


def run_non_llm_parallel(
    instance:       str,
    nodes:          np.ndarray,
    distances:      np.ndarray,
    seeds:          list,
    initial_routes: dict,
    cache_data:     dict,
) -> dict:
    """
    [EXP-6] 用字典按 seed 存储结果，最后按 SEEDS 顺序组装，
    消除 as_completed 乱序导致的种子错位问题。
    """
    seed_offset = {"baseline": 0, "traditional_alns": 1000}

    store = {s: {} for s in NON_LLM_STRATEGIES}
    tasks = []

    for strategy in NON_LLM_STRATEGIES:
        if strategy not in cache_data:
            cache_data[strategy] = {}
        for seed in seeds:
            s_str = str(seed)
            if s_str in cache_data[strategy]:
                c = cache_data[strategy][s_str]
                store[strategy][seed] = (c["history"], c["best_dist"], c["best_route"])
                print(f"   ⏭️ [{strategy:<18}] seed={seed} (缓存命中)")
            else:
                tasks.append((instance, nodes, distances,
                              strategy,
                              seed + seed_offset[strategy],   # solver_seed
                              seed,                           # orig_seed
                              initial_routes[seed]))

    n_workers = min(MAX_PARALLEL_WORKERS, len(tasks)) if tasks else 0
    if tasks:
        print(f"\n⚡ [并行] 启动 {n_workers} 个进程，共 {len(tasks)} 个非LLM任务...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_non_llm, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    strategy, orig_seed, history, best_dist, best_route = future.result()
                    store[strategy][orig_seed] = (history, best_dist, best_route)
                    print(f"   ✅ [{strategy:<18}] seed={orig_seed} "
                          f"完成，最优距离: {best_dist:.1f}")
                    
                    # 存入缓存
                    cache_data[strategy][str(orig_seed)] = {
                        "history": history,
                        "best_dist": best_dist,
                        "best_route": best_route
                    }
                    save_cache(instance, cache_data)
                except Exception as e:
                    task = futures[future]
                    print(f"   ❌ 任务失败 [{task[3]}, seed={task[6]}]: {e}")

    # [EXP-6] 按 SEEDS 顺序组装，保证与 sc_llm_os 结果一一对应
    results = {}
    for strategy in NON_LLM_STRATEGIES:
        results[strategy] = {
            "histories": [store[strategy][s][0] for s in seeds
                          if s in store[strategy]],
            "bests":     [store[strategy][s][1] for s in seeds
                          if s in store[strategy]],
            "routes":    [store[strategy][s][2] for s in seeds
                          if s in store[strategy]],
        }
    return results


def run_llm_serial(
    instance:       str,
    nodes:          np.ndarray,
    distances:      np.ndarray,
    seeds:          list,
    initial_routes: dict,
    cache_data:     dict,
) -> dict:
    """
    sc_llm_os 串行执行（API 限流保护）。
    自动隔离含 Fallback 的脏数据 seed。
    """
    results = {
        "histories":   [],
        "bests":       [],
        "routes":      [],
        "dirty_seeds": [],
        "clean_bests": [],
        "llm_logs":    {},
    }
    
    if "sc_llm_os" not in cache_data:
        cache_data["sc_llm_os"] = {}

    for seed in seeds:
        s_str = str(seed)
        if s_str in cache_data["sc_llm_os"]:
            c = cache_data["sc_llm_os"][s_str]
            history = c["history"]
            best_dist = c["best_dist"]
            best_route = c["best_route"]
            llm_log = c.get("llm_log", [])
            has_dirty = c.get("has_dirty", False)
            print(f"\n   ⏭️ [sc_llm_os] seed={seed} (缓存命中)")
        else:
            solver_seed = seed + 2000
            print(f"\n   [sc_llm_os] seed={seed} (solver_seed={solver_seed}) 启动...")

            history, best_dist, best_route, llm_log, has_dirty = run_solver(
                nodes         = nodes,
                distances     = distances,
                initial_route = initial_routes[seed],
                strategy      = "sc_llm_os",
                solver_seed   = solver_seed,
            )
            # 存入缓存
            cache_data["sc_llm_os"][s_str] = {
                "history": history,
                "best_dist": best_dist,
                "best_route": best_route,
                "llm_log": llm_log,
                "has_dirty": has_dirty
            }
            save_cache(instance, cache_data)

        results["histories"].append(history)
        results["bests"].append(best_dist)
        results["routes"].append(best_route)
        results["llm_logs"][seed] = llm_log

        if has_dirty:
            results["dirty_seeds"].append(seed)
            print(f"   ⚠️  seed={seed} 含Fallback，加入脏数据列表。")
        else:
            results["clean_bests"].append(best_dist)

        if llm_log:
            log_path = os.path.join(
                OUTPUT_DIR, f"llm_log_{instance}_seed{seed}_{RUN_ID}.json"
            )
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(llm_log, f, indent=2, ensure_ascii=False)
            print(f"   💾 LLM日志已保存: {log_path}")

    if results["dirty_seeds"]:
        print(f"\n   🔴 [数据质量警告] 脏数据seed: {results['dirty_seeds']}")
        print(f"   📊 清洁样本: {len(results['clean_bests'])} / {len(seeds)}")
        if len(results["clean_bests"]) < 3:
            print("   ⚠️  清洁样本不足3个，建议检查网络后重跑。")

    return results


# =============================================================================
# 模块九：可视化
# =============================================================================

def plot_results(
    all_histories: dict,
    final_results: dict,
    instance_name: str,
    llm_logs:      list = None,
):
    """
    双图布局：
    左图 — 多种子均值收敛曲线 + 标准差阴影 + LLM触发时间线
    右图 — 最终解质量箱线图
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"SC-LLM-OS Ablation Study  |  {instance_name}  "
        f"|  Seeds={SEEDS}  |  2-opt={'ON' if USE_TWO_OPT else 'OFF'}",
        fontsize=12, fontweight="bold",
    )

    style_map = {
        "baseline":         ("gray",       "--",  "Baseline (Uniform)"),
        "traditional_alns": ("darkorange", "-.",  "Traditional ALNS"),
        "sc_llm_os":        ("royalblue",  "-",   "SC-LLM-OS (Ours)"),
    }

    # ── 左图：均值收敛曲线 ──────────────────────────────────────────────────
    ax = axes[0]
    for strategy in STRATEGIES:
        hists = all_histories.get(strategy, [])
        if not hists:
            continue
        arr        = np.array(hists)
        mean_curve = arr.mean(axis=0)
        std_curve  = arr.std(axis=0)
        color, ls, label = style_map[strategy]
        x        = np.arange(len(mean_curve))
        best_val = float(np.min(arr))
        ax.plot(mean_curve,
                label=f"{label}  (best={best_val:.1f})",
                color=color, linestyle=ls, linewidth=2.2, zorder=3)
        ax.fill_between(x,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.12, color=color, zorder=2)

    # LLM 触发时间线标注
    if llm_logs:
        mode_color = {
            "exploit":          "#27ae60",
            "explore_topology": "#e67e22",
            "explore_global":   "#e74c3c",
        }
        plotted_modes = set()
        for entry in llm_logs:
            is_fb = entry.get("is_fallback", False)
            mc    = mode_color.get(entry["mode"], "purple")
            lbl   = None
            if entry["mode"] not in plotted_modes and not is_fb:
                lbl = f"LLM: {entry['mode']}"
                plotted_modes.add(entry["mode"])
            ax.axvline(x        = entry["iteration"],
                       color    = mc,
                       alpha    = 0.30,
                       linewidth= 1.4 if is_fb else 0.8,
                       linestyle= ":" if is_fb else "-",
                       label    = lbl,
                       zorder   = 1)

    ax.set_title("Mean Convergence Curve ± Std Dev", fontsize=11)
    ax.set_xlabel("Iterations", fontsize=10)
    ax.set_ylabel("Best Distance Found", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # ── 右图：箱线图 ────────────────────────────────────────────────────────
    ax2        = axes[1]
    box_data   = [final_results.get(s, []) for s in STRATEGIES]
    box_labels = ["Baseline", "Trad-ALNS", "SC-LLM-OS"]
    colors_box = ["gray", "darkorange", "royalblue"]

    bp = ax2.boxplot(
        box_data,
        labels       = box_labels,
        patch_artist = True,
        widths       = 0.5,
        medianprops  = dict(color="white", linewidth=2.0),
        whiskerprops = dict(linewidth=1.4),
        capprops     = dict(linewidth=1.4),
        flierprops   = dict(marker="o", markersize=5, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, (strategy, color) in enumerate(zip(STRATEGIES, colors_box), start=1):
        data = final_results.get(strategy, [])
        if data:
            mv = float(np.mean(data))
            ax2.text(i, mv, f"  μ={mv:.1f}",
                     fontsize=8, color=color,
                     va="center", fontweight="bold")

    dirty = final_results.get("sc_llm_os_dirty_seeds", [])
    title_suffix = (f"\n⚠️ SC-LLM-OS: {len(dirty)} dirty seed(s)"
                    if dirty else "")
    ax2.set_title(f"Final Solution Quality{title_suffix}", fontsize=11)
    ax2.set_ylabel("Best Distance Found", fontsize=10)
    ax2.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"results_{instance_name}_{RUN_ID}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n   📊 收敛图已保存: {save_path}")
    plt.close()


def plot_llm_decision_timeline(
    llm_logs:      list,
    instance_name: str,
    seed:          int,
):
    """
    LLM 决策时间线图（答辩展示用）。
    正常决策用彩色符号标注，Fallback 用红色 × 标注。
    """
    if not llm_logs:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    mode_color  = {
        "exploit":          "#27ae60",
        "explore_topology": "#e67e22",
        "explore_global":   "#c0392b",
    }
    mode_marker = {
        "exploit":          "^",
        "explore_topology": "s",
        "explore_global":   "D",
    }

    iterations = [e["iteration"]     for e in llm_logs]
    best_dists = [e["best_distance"]  for e in llm_logs]

    ax.plot(iterations, best_dists,
            color="royalblue", linewidth=1.5, zorder=2,
            label="Best Distance at Trigger")

    for entry in llm_logs:
        it     = entry["iteration"]
        bd     = entry["best_distance"]
        mode   = entry["mode"]
        reason = entry.get("reasoning", "")
        is_fb  = entry.get("is_fallback", False)

        c  = mode_color.get(mode, "gray")
        mk = "x" if is_fb else mode_marker.get(mode, "o")
        sz = 130 if is_fb else 80
        ax.scatter(it, bd, color="red" if is_fb else c,
                   marker=mk, s=sz, zorder=3,
                   linewidths=2.5 if is_fb else 1.0)

        text = (f"[FB]\n{mode}" if is_fb
                else (f"{mode}\n{reason[:26]}…"
                      if len(reason) > 26 else f"{mode}\n{reason}"))
        ax.annotate(
            text,
            xy         = (it, bd),
            xytext     = (0, 16),
            textcoords = "offset points",
            fontsize   = 6,
            color      = "red" if is_fb else c,
            ha         = "center",
            arrowprops = dict(arrowstyle="-",
                              color="red" if is_fb else c,
                              lw=0.6),
        )

    legend_patches = [
        mpatches.Patch(color=mode_color["exploit"],
                       label="exploit"),
        mpatches.Patch(color=mode_color["explore_topology"],
                       label="explore_topology"),
        mpatches.Patch(color=mode_color["explore_global"],
                       label="explore_global"),
        mpatches.Patch(color="red",
                       label="FALLBACK (dirty)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
    ax.set_title(
        f"LLM Decision Timeline  |  {instance_name}  |  Seed={seed}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Iteration",                  fontsize=10)
    ax.set_ylabel("Best Distance at Trigger",   fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR, f"llm_timeline_{instance_name}_seed{seed}_{RUN_ID}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   🧠 LLM时间线图已保存: {save_path}")
    plt.close()


# =============================================================================
# 模块十：结果持久化
# =============================================================================

def run_significance_tests(final_results: dict) -> dict:
    """对 SC-LLM-OS vs 竞争对手做 Wilcoxon 秩和检验"""
    test_results = {}
    if "sc_llm_os" not in final_results or len(final_results["sc_llm_os"]) == 0:
        return test_results
        
    sc_data = np.array(final_results["sc_llm_os"])
    for competitor in ["baseline", "traditional_alns"]:
        if competitor in final_results and len(final_results[competitor]) > 0:
            comp_data = np.array(final_results[competitor])
            try:
                # 如果完全一样，stats.wilcoxon 会报错，特殊处理
                if np.all(sc_data == comp_data):
                    stat, p_value = 0.0, 1.0
                else:
                    stat, p_value = stats.wilcoxon(sc_data, comp_data)
                test_results[f"sc_llm_os_vs_{competitor}"] = {
                    "statistic": round(float(stat), 4),
                    "p_value":   round(float(p_value), 4),
                    "significant_at_0.05": bool(p_value < 0.05),
                }
            except Exception as e:
                print(f"   ⚠️ Wilcoxon 检验出现异常 ({competitor}): {e}")
    return test_results

def save_results(
    final_results: dict,
    all_histories: dict,
    llm_logs_all:  dict,
    dirty_seeds:   list,
    instance_name: str,
    num_iterations: int,
    stagnation_reset_threshold: int,
):
    payload = {
        "instance": instance_name,
        "seeds":    SEEDS,
        "config": {
            "NUM_ITERATIONS":             num_iterations,
            "DESTRUCTION_RATIO":          DESTRUCTION_RATIO,
            "USE_TWO_OPT":                USE_TWO_OPT,
            "MAX_TWO_OPT_PASSES":         MAX_TWO_OPT_PASSES,
            "USE_ELITE_RESTART":          USE_ELITE_RESTART,
            "LLM_TRIGGER_ONLY_ON_STAGNATION": LLM_TRIGGER_ONLY_ON_STAGNATION,
            "ALNS_RHO":                   ALNS_RHO,
            "BASE_PROBS":                 BASE_PROBS,
            "MIN_PROB":                   MIN_PROB,
            "STAGNATION_RESET_THRESHOLD": stagnation_reset_threshold,
        },
        "data_quality": {
            "dirty_seeds": dirty_seeds,
            "n_clean":     len(SEEDS) - len(dirty_seeds),
            "n_total":     len(SEEDS),
        },
        "summary":              {},
        "significance_tests":   run_significance_tests(final_results),
        "all_final_distances":  {
            s: [round(float(v), 2) for v in vals]
            for s, vals in final_results.items()
            if isinstance(vals, list)
        },
        "llm_logs": {str(k): v for k, v in llm_logs_all.items()},
    }

    for strategy in STRATEGIES:
        data = final_results.get(strategy, [])
        if not data:
            continue
        entry = {
            "mean": round(float(np.mean(data)), 2),
            "std":  round(float(np.std(data)),  2),
            "min":  round(float(np.min(data)),  2),
            "max":  round(float(np.max(data)),  2),
        }
        if strategy == "sc_llm_os":
            clean = final_results.get("sc_llm_os_clean", [])
            entry["clean_mean"] = round(float(np.mean(clean)), 2) if clean else None
            entry["clean_std"]  = round(float(np.std(clean)),  2) if clean else None
            entry["clean_n"]    = len(clean)
        payload["summary"][strategy] = entry

    result_path = os.path.join(OUTPUT_DIR, f"results_{instance_name}_{RUN_ID}_{LLM_PROVIDER}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"   💾 实验结果已保存: {result_path}")

    history_path = os.path.join(OUTPUT_DIR, f"histories_{instance_name}_{RUN_ID}_{LLM_PROVIDER}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {s: [list(map(float, h)) for h in hs]
             for s, hs in all_histories.items()},
            f, ensure_ascii=False,
        )
    print(f"   💾 收敛历史已保存: {history_path}")


# =============================================================================
# 模块十一：统计表格打印
# =============================================================================

def print_stats_table(
    final_results: dict,
    clean_bests:   list,
    dirty_seeds:   list,
    instance_name: str,
):
    width = 70
    print("\n" + "=" * width)
    print(f"{'实验统计：' + instance_name + '  Seeds=' + str(SEEDS):^{width}}")
    print("=" * width)
    print(f"  {'Strategy':<22} | {'Mean':>9} | {'Std':>7} "
          f"| {'Min':>9} | {'Max':>9}")
    print("-" * width)

    means = {}
    for strategy in STRATEGIES:
        data = final_results.get(strategy, [])
        if not data:
            continue
        m, s = float(np.mean(data)), float(np.std(data))
        mn,mx= float(np.min(data)),  float(np.max(data))
        means[strategy] = m
        tag = " ⚠️" if (strategy == "sc_llm_os" and dirty_seeds) else ""
        print(f"  {strategy + tag:<22} | {m:>9.1f} | {s:>7.1f} "
              f"| {mn:>9.1f} | {mx:>9.1f}")

    if clean_bests and dirty_seeds:
        cm, cs = float(np.mean(clean_bests)), float(np.std(clean_bests))
        cmn,cmx= float(np.min(clean_bests)), float(np.max(clean_bests))
        print(f"  {'sc_llm_os (clean)':<22} | {cm:>9.1f} | {cs:>7.1f} "
              f"| {cmn:>9.1f} | {cmx:>9.1f}  "
              f"(n={len(clean_bests)}/{len(SEEDS)})")

    if means:
        best_s = min(means, key=means.get)
        print("-" * width)
        print(f"  🏆 最优策略: {best_s}  (Mean={means[best_s]:.1f})")

    sig_tests = run_significance_tests(final_results)
    if sig_tests:
        print("-" * width)
        print(f"  [Wilcoxon 秩和检验 (SC-LLM-OS vs 其他)]")
        for test_name, v in sig_tests.items():
            sig_mark = "✅ 显著" if v["significant_at_0.05"] else "❌ 不显著"
            print(f"    {test_name:<30}: p={v['p_value']:<6.4f} ({sig_mark})")

    if dirty_seeds:
        print("-" * width)
        print(f"  ⚠️  脏数据seed: {dirty_seeds}（Fallback触发，clean版本已剔除）")

    print("=" * width)


# =============================================================================
# 模块十二：主程序入口
# =============================================================================

if __name__ == "__main__":

    if LLM_PROVIDER == "gemini":
        if not API_KEY or not API_KEY.strip():
            raise SystemExit(
                "\n❌ [Fail-fast] GEMINI_API_KEY 未设置！\n"
                "   请通过环境变量设置：\n"
                "   Windows CMD : set GEMINI_API_KEY=你的key\n"
                "   或直接在脚本顶部 API_KEY = '' 中填入（仅本地调试）"
            )
        print("✅ Gemini API Key 校验通过。")
    elif LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY or not DEEPSEEK_API_KEY.strip() or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
            raise SystemExit(
                "\n❌ [Fail-fast] DEEPSEEK_API_KEY 未设置！\n"
                "   请通过环境变量设置，或直接在脚本顶部填入（仅本地调试）"
            )
        print("✅ DeepSeek API Key 校验通过。")
    else:
        raise SystemExit(f"\n❌ [Fail-fast] 未知的 LLM_PROVIDER: {LLM_PROVIDER}")

    print("\n" + "=" * 60)
    print("   SC-LLM-OS  v5.0  |  消融实验启动")
    print(f"   LLM: {LLM_PROVIDER}")
    print(f"   实例    : {INSTANCES}")
    print(f"   种子    : {SEEDS}")
    print(f"   策略    : {STRATEGIES}")
    print(f"   破坏比例: {DESTRUCTION_RATIO}")
    print(f"   2-opt   : {'开启' if USE_TWO_OPT else '关闭'}")
    print(f"   精英重启: {'开启' if USE_ELITE_RESTART else '关闭'}")
    print(f"   仅应急触发: {'开启' if LLM_TRIGGER_ONLY_ON_STAGNATION else '关闭'}")
    print(f"   并行数  : {MAX_PARALLEL_WORKERS} (仅非LLM策略)")
    print("=" * 60)

    for instance in INSTANCES:

        # ── 数据加载 ────────────────────────────────────────────────────────
        nodes, distances = load_tsplib_data(instance)
        if nodes is None:
            print(f"⚠️  跳过 {instance}（数据加载失败）")
            continue
        num_cities = len(nodes)
        
        # 动态计算并应用触发间隔、破坏比例
        num_cities = len(nodes)
        NUM_ITERATIONS = 200 + num_cities * 5
        STAGNATION_RESET_THRESHOLD = max(30, int(num_cities * 0.3))
        llm_trigger_interval = calc_trigger_interval(num_cities, NUM_ITERATIONS)
        k_destroy = max(3, int(num_cities * DESTRUCTION_RATIO))
        
        print(f"\n{'='*60}")
        print(f"  实例: {instance}  ({num_cities} 城市)")
        print(f"  动态触发间隔: {llm_trigger_interval} (n={num_cities})")
        print(f"  动态破坏规模: {k_destroy} (n={num_cities})")
        print(f"{'='*60}")

        # ── 初始解生成（独立RNG，与求解种子隔离）────────────────────────────
        # 问题3修复：为每个seed生成唯一的初始解，所有策略共享，实现公平比较
        initial_routes = {}
        for seed in SEEDS:
            init_rng = np.random.RandomState(seed)
            start = int(init_rng.randint(0, num_cities))
            initial_routes[seed] = greedy_initial_route(num_cities, distances, start)

        # ── 加载本实例的断点缓存 ──────────────────────────────────────────
        cache_data = load_cache(instance)

        # ── 结果容器 ─────────────────────────────────────────────────────────
        final_results = {s: [] for s in STRATEGIES}
        all_histories = {s: [] for s in STRATEGIES}
        all_routes    = {s: [] for s in STRATEGIES}

        # ── 第一阶段：并行跑非LLM策略 ───────────────────────────────────────
        print(f"\n📌 第一阶段：并行运行 {NON_LLM_STRATEGIES}...")
        non_llm_res = run_non_llm_parallel(
            instance       = instance,
            nodes          = nodes,
            distances      = distances,
            seeds          = SEEDS,
            initial_routes = initial_routes,
            cache_data     = cache_data,
        )
        for strategy in NON_LLM_STRATEGIES:
            final_results[strategy] = non_llm_res[strategy]["bests"]
            all_histories[strategy] = non_llm_res[strategy]["histories"]
            all_routes[strategy]    = non_llm_res[strategy]["routes"]

        # ── 第二阶段：串行跑 SC-LLM-OS ──────────────────────────────────────
        print(f"\n📌 第二阶段：串行运行 SC-LLM-OS...")
        llm_res = run_llm_serial(
            instance       = instance,
            nodes          = nodes,
            distances      = distances,
            seeds          = SEEDS,
            initial_routes = initial_routes,
            cache_data     = cache_data,
        )
        final_results["sc_llm_os"]            = llm_res["bests"]
        final_results["sc_llm_os_clean"]      = llm_res["clean_bests"]
        final_results["sc_llm_os_dirty_seeds"]= llm_res["dirty_seeds"]
        all_histories["sc_llm_os"]            = llm_res["histories"]
        all_routes["sc_llm_os"]               = llm_res["routes"]
        dirty_seeds                           = llm_res["dirty_seeds"]
        llm_logs_all                          = llm_res["llm_logs"]

        # ── 独立存储每个种子的解 ────────────────────────────────────────────
        for i, seed in enumerate(SEEDS):
            seed_dir = os.path.join(OUTPUT_DIR, instance, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            for strategy in STRATEGIES:
                if len(all_routes.get(strategy, [])) > i:
                    route_path = os.path.join(seed_dir, f"route_{strategy}.json")
                    with open(route_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "strategy": strategy,
                            "seed": seed,
                            "distance": float(final_results[strategy][i]),
                            "route": [int(node) for node in all_routes[strategy][i]]
                        }, f, indent=2)
        print(f"   💾 各种子路线单独保存完毕 ({instance})")

        # ── 统计表格 ────────────────────────────────────────────────────────
        print_stats_table(
            final_results = final_results,
            clean_bests   = llm_res["clean_bests"],
            dirty_seeds   = dirty_seeds,
            instance_name = instance,
        )

        # ── 持久化 ──────────────────────────────────────────────────────────
        print(f"\n📌 第三阶段：保存结果...")
        save_results(
            final_results = final_results,
            all_histories = all_histories,
            llm_logs_all  = llm_logs_all,
            dirty_seeds   = dirty_seeds,
            instance_name = instance,
            num_iterations= NUM_ITERATIONS,
            stagnation_reset_threshold= STAGNATION_RESET_THRESHOLD,
        )

        # ── 可视化 ──────────────────────────────────────────────────────────
        print(f"\n📌 第四阶段：生成图表...")
        plot_results(
            all_histories = all_histories,
            final_results = final_results,
            instance_name = instance,
            llm_logs      = llm_logs_all.get(SEEDS[-1], []),
        )
        for seed in SEEDS:
            if llm_logs_all.get(seed):
                plot_llm_decision_timeline(
                    llm_logs      = llm_logs_all[seed],
                    instance_name = instance,
                    seed          = seed,
                )

    print("\n" + "=" * 60)
    print(f"   🎉  所有实验完成！结果保存于: {OUTPUT_DIR}")
    print("=" * 60)