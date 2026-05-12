# =============================================================================
#  模块十：结果持久化
#  模块十一：统计表格打印
# =============================================================================

import os
import json

import numpy as np
from scipy import stats as scipy_stats

from .config import (
    SEEDS,
    STRATEGIES,
    DESTRUCTION_RATIO,
    USE_TWO_OPT,
    MAX_TWO_OPT_PASSES,
    USE_ELITE_RESTART,
    LLM_TRIGGER_ONLY_ON_STAGNATION,
    ALNS_RHO,
    BASE_PROBS,
    MIN_PROB,
    OUTPUT_DIR,
    RUN_ID,
)


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
                    stat, p_value = scipy_stats.wilcoxon(sc_data, comp_data)
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
    LLM_PROVIDER: str,
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
