# =============================================================================
#  模块五：状态提取 S2N（State → Natural-language JSON）
# =============================================================================

import json

import numpy as np


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
