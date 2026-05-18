# =============================================================================
#  模块十二：主程序入口
# =============================================================================

import os
import sys
import json

import numpy as np

from .config import (
    LLM_PROVIDER,
    API_KEY,
    DEEPSEEK_API_KEY,
    MIMO_API_KEY,
    INSTANCES,
    SEEDS,
    STRATEGIES,
    NON_LLM_STRATEGIES,
    DESTRUCTION_RATIO,
    DESTROY_OPS,
    REPAIR_OPS,
    USE_TWO_OPT,
    USE_ELITE_RESTART,
    LLM_TRIGGER_ONLY_ON_STAGNATION,
    MAX_PARALLEL_WORKERS,
    OUTPUT_DIR,
    RUN_ID,
    VERSION,
    load_cache,
)
from .utils import calc_trigger_interval
from .data_loader import load_tsplib_data
from .operators import greedy_initial_route
from .runner import run_non_llm_parallel, run_llm_serial
from .stats import print_stats_table, save_results
from .visualize import plot_results, plot_llm_decision_timeline


def main():
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
    elif LLM_PROVIDER == "mimo":
        if not MIMO_API_KEY or not MIMO_API_KEY.strip() or MIMO_API_KEY == "your_mimo_api_key_here":
            raise SystemExit(
                "\n❌ [Fail-fast] MIMO_API_KEY 未设置！\n"
                "   请通过环境变量设置：\n"
                "   Windows CMD : set MIMO_API_KEY=你的key\n"
                "   或在 .env 文件中填入"
            )
        print("✅ Xiaomi Mimo API Key 校验通过。")
    else:
        raise SystemExit(f"\n❌ [Fail-fast] 未知的 LLM_PROVIDER: {LLM_PROVIDER}")

    print("\n" + "=" * 60)
    print(f"   SC-LLM-OS  {VERSION}  |  消融实验启动")
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
        NUM_ITERATIONS = 200 + num_cities * 5
        STAGNATION_RESET_THRESHOLD = max(30, int(num_cities * 0.3))
        llm_trigger_interval = calc_trigger_interval(
            num_cities,
            NUM_ITERATIONS,
            max(len(DESTROY_OPS), len(REPAIR_OPS)),
        )
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
        all_decision_logs = {s: {} for s in STRATEGIES}

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
            all_decision_logs[strategy] = non_llm_res[strategy].get("decision_logs", {})

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
        all_decision_logs["sc_llm_os"]        = llm_logs_all

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
            LLM_PROVIDER = LLM_PROVIDER,
            decision_logs_all = all_decision_logs,
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


if __name__ == "__main__":
    main()
