# =============================================================================
#  模块八：并行加速（分层策略）
#  - 非LLM策略并行执行
#  - SC-LLM-OS 串行执行（API 限流保护）
# =============================================================================

import os
import json

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import (
    NON_LLM_STRATEGIES,
    MAX_PARALLEL_WORKERS,
    OUTPUT_DIR,
    RUN_ID,
    load_cache,
    save_cache,
)
from .solver import run_solver


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
                    print(f"   ❌ 任务失败 [{task[3]}, seed={task[5]}]: {e}")

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
