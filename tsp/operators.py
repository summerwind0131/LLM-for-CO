# =============================================================================
#  模块四：算子库
#  - 2-opt 局部搜索
#  - 贪心初始解构造
#  - 破坏算子 (random / worst / segment)
#  - 修复算子 (greedy)
#  - 算子注册表
# =============================================================================

import random

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from .config import MAX_TWO_OPT_PASSES


# ── 2-opt 局部搜索 ────────────────────────────────────────────────────────────

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


# ── 贪心初始解 ────────────────────────────────────────────────────────────────

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
