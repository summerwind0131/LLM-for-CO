# =============================================================================
#  模块三：工具函数
#  - 距离计算（含 Numba JIT 加速）
#  - 轮盘赌选择
#  - 动态触发间隔计算
# =============================================================================

import random

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ── 距离计算 ──────────────────────────────────────────────────────────────────

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


# ── 轮盘赌选择 ────────────────────────────────────────────────────────────────

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


# ── 动态触发间隔 ──────────────────────────────────────────────────────────────

def calc_trigger_interval(num_cities: int, num_iterations: int, n_ops: int = 3) -> int:
    """
    动态触发间隔：保证每次触发时每种算子至少有 MIN_SAMPLES_PER_OP 次使用
    从而保证统计可靠性不随规模下降。

    公式：interval = MIN_SAMPLES_PER_OP × n_ops × scale_factor
    n_ops 默认为 3，调用方可传入当前算子族的最大算子数，
    如 DESTROY_OPS 数量变化请通过参数传入。
    """
    MIN_SAMPLES_PER_OP = 15      # 每种算子最少样本量

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
