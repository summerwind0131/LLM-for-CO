# =============================================================================
# SC-LLM-OS v8.0 — CVRP/Solomon Edition
# Strategic Commander LLM for Operator Selection in ALNS
#
# 对标: "Graph RL for Operator Selection in ALNS" (Johnn et al., 2023)
#
# [C-1] 问题域: TSP → CVRP (Solomon R/C/RC benchmark)
# [C-2] 算子库: 3+1 → 12 destroy + 2 repair (完全对齐论文)
# [C-3] 新增 MDP 独立评估框架 (累积奖励指标, 对标 Table 1)
# [C-4] 新增 Portfolio 规模扫描 (|D|=2..12, 对标 Table 2)
# [C-5] 新增 Destroy Scale 扫描 (d∈{2,4,6,8,10})
# [C-6] 新增 LRW (Learned Roulette Wheel) 基线
# [C-7] 评估指标: avg_obj + best_obj (对标论文 Table 2 格式)
# =============================================================================

import numpy as np
import random
import json
import urllib.request
import os
import sys
import time
import math
import platform
import re
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# 模块一：全局配置
# =============================================================================

LLM_PROVIDER     = os.environ.get("LLM_PROVIDER", "deepseek").lower()
API_KEY          = os.environ.get("GEMINI_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
CUHK_API_KEY     = os.environ.get("CUHK_API_KEY", "").strip()
CUHK_MODEL_NAME  = os.environ.get("CUHK_MODEL_NAME", "").strip()

# ── 实验配置（与论文对齐）─────────────────────────────────────────────────────
# Solomon 三类实例各选1个代表
SOLOMON_INSTANCES  = ["R101", "C101", "RC101"]      # R / C / RC 三类
SEEDS              = list(range(0, 10))              # 10 seeds（与论文一致）
N_INIT_SOLUTIONS   = 128                             # 每实例128个初始解（与论文一致）

# 论文完整算子库（按顺序加入）
DESTROY_OPS_ALL = [
    "random_node",          # 1  random-based
    "random_route",         # 2  random-based
    "worst_node",           # 3  greedy-based
    "neighbourhood",        # 4  greedy-based
    "greedy_route",         # 5  greedy-based
    "proximity",            # 6  related-based (Shaw)
    "cluster",              # 7  related-based
    "node_neighbourhood",   # 8  related-based
    "zone",                 # 9  related-based
    "route_neighbourhood",  # 10 related-based
    "pair",                 # 11 related-based
    "historical_pair",      # 12 related-based
]
REPAIR_OPS_ALL = ["greedy", "two_regret"]

# ── MDP 评估参数（与论文对齐）────────────────────────────────────────────────
MDP_BUDGET         = 10      # operator pair budget b=10
MDP_DESTROY_SCALE  = 4       # 默认 destroy scale d=4（n=20时）
MDP_SMALL_N        = 20      # 训练用小规模（前20个客户节点）

# ── 算法超参数 ────────────────────────────────────────────────────────────────
ALNS_RHO           = 0.5     # 轮盘赌平滑系数
MIN_PROB           = 0.1     # 探索保底
EPSILON            = 1e-8
BASE_PROBS_ALL     = None    # 动态生成，均匀分布

# ── Sweep 参数 ────────────────────────────────────────────────────────────────
PORTFOLIO_SIZES    = list(range(2, 13))              # |D| = 2..12
DESTROY_SCALES     = [2, 4, 6, 8, 10]               # destroy scale sweep

# ── 模块开关 ─────────────────────────────────────────────────────────────────
RUN_RL_MODULE      = True     # [开关] 设为 True 则运行强化学习(LRW)模块
RUN_LLM_MODULE     = True     # [开关] 设为 True 则运行大模型(LLM)决策模块

# ── 策略标签 ─────────────────────────────────────────────────────────────────
STRATEGIES         = ["random"]
if RUN_RL_MODULE:
    STRATEGIES.append("lrw")
if RUN_LLM_MODULE:
    STRATEGIES.append("sc_llm_os")

# ── 输出目录 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION    = "v8.0"
RUN_ID     = f"{VERSION}_{time.strftime('%Y%m%d_%H%M%S')}_{LLM_PROVIDER}"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sc_llm_os_cvrp_results", RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 模块二：Solomon CVRP 数据加载
# =============================================================================

class CVRPInstance:
    """存储一个 Solomon CVRP 实例"""
    def __init__(self, name, coords, demands, capacity, dist_matrix):
        self.name        = name
        self.coords      = coords          # shape (n+1, 2), 索引0为depot
        self.demands     = demands         # shape (n+1,), demands[0]=0
        self.capacity    = capacity
        self.dist_matrix = dist_matrix     # shape (n+1, n+1)
        self.n_customers = len(coords) - 1 # 不含depot

    def route_cost(self, route: list) -> float:
        """单条路线的总距离（depot→c1→...→cn→depot）"""
        if not route:
            return 0.0
        d   = self.dist_matrix
        cost = d[0][route[0]]
        for i in range(len(route) - 1):
            cost += d[route[i]][route[i + 1]]
        cost += d[route[-1]][0]
        return cost

    def solution_cost(self, solution: list) -> float:
        """多路线解的总距离"""
        return sum(self.route_cost(r) for r in solution)

    def route_load(self, route: list) -> int:
        return sum(self.demands[c] for c in route)

    def is_feasible(self, solution: list) -> bool:
        all_customers = set()
        for route in solution:
            if self.route_load(route) > self.capacity:
                return False
            all_customers.update(route)
        return len(all_customers) == self.n_customers


def load_solomon(instance_name: str, max_customers: int = None) -> CVRPInstance:
    """
    加载 Solomon 实例（本地没有则自动下载）。
    max_customers: 截取前N个客户（用于MDP小规模训练）。
    """
    print(f"\n📦 加载 Solomon 实例: {instance_name} ...")
    data_dir  = os.path.join(SCRIPT_DIR, "data", "solomon")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{instance_name}.txt")

    if not os.path.exists(file_path):
        # 尝试多个镜像源
        urls = [
            f"https://www.sintef.no/contentassets/db44a7ebe2a540e5a3e6b5c3b1cc5e5e/solomon/{instance_name}.txt",
            f"http://web.cba.neu.edu/~msolomon/problems/{instance_name.lower()}.txt",
        ]
        downloaded = False
        for url in urls:
            try:
                print(f"   🌐 尝试下载: {url}")
                urllib.request.urlretrieve(url, file_path)
                downloaded = True
                print("   ✅ 下载成功")
                break
            except Exception as e:
                print(f"   ⚠️ 失败: {e}")
        if not downloaded:
            # 生成合成Solomon实例用于调试
            print(f"   🔧 生成合成Solomon实例: {instance_name}")
            _generate_synthetic_solomon(file_path, instance_name, n=100)

    return _parse_solomon_file(file_path, instance_name, max_customers)


def _generate_synthetic_solomon(file_path: str, name: str, n: int = 100):
    """生成合成Solomon格式文件（下载失败时的fallback）"""
    rng = np.random.RandomState(hash(name) % 10000)

    # 根据实例名决定分布类型
    itype = name[0].upper()
    if itype == "C":       # clustered
        n_clusters = 5
        centers    = rng.rand(n_clusters, 2) * 80 + 10
        coords     = []
        for i in range(n):
            c = centers[i % n_clusters]
            coords.append(c + rng.randn(2) * 5)
        coords = np.clip(coords, 1, 99)
    elif itype == "R":     # random
        coords = rng.rand(n, 2) * 90 + 5
    else:                  # RC: mixed
        half = n // 2
        centers = rng.rand(3, 2) * 80 + 10
        c1 = np.clip(np.array([centers[i % 3] + rng.randn(2) * 8
                                for i in range(half)]), 1, 99)
        c2 = rng.rand(n - half, 2) * 90 + 5
        coords = np.vstack([c1, c2])

    depot  = np.array([[50.0, 50.0]])
    all_coords = np.vstack([depot, coords])
    demands    = rng.randint(5, 25, size=n + 1)
    demands[0] = 0
    capacity   = 200

    with open(file_path, "w") as f:
        f.write(f"NAME : {name}\nCOMMENT : Synthetic\nTYPE : CVRP\n")
        f.write(f"VEHICLE\nNUMBER\tCAPACITY\n999\t{capacity}\n\n")
        f.write("CUSTOMER\n")
        f.write("CUST NO.\tXCOORD.\tYCOORD.\tDEMAND\t"
                "READY TIME\tDUE DATE\tSERVICE TIME\n")
        for i in range(n + 1):
            f.write(f"{i}\t{all_coords[i,0]:.1f}\t{all_coords[i,1]:.1f}\t"
                    f"{demands[i]}\t0\t1000\t10\n")
    print(f"   ✅ 合成实例已保存: {file_path}")


def _parse_solomon_file(file_path: str, name: str,
                        max_customers: int = None) -> CVRPInstance:
    with open(file_path, "r") as f:
        lines = f.readlines()

    capacity   = 200
    customers  = []    # [(id, x, y, demand), ...]
    in_customer = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 解析容量
        if re.match(r"^\d+\s+\d+", line):
            parts = line.split()
            if len(parts) == 2:
                try:
                    capacity = int(parts[1])
                except:
                    pass
        # 解析客户节点
        if in_customer:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    cid    = int(parts[0])
                    x, y   = float(parts[1]), float(parts[2])
                    demand = int(parts[3])
                    customers.append((cid, x, y, demand))
                except:
                    pass
        if "CUST" in line.upper() and "NO" in line.upper():
            in_customer = True

    if not customers:
        raise ValueError(f"无法解析Solomon文件: {file_path}")

    # 截取规模
    if max_customers is not None:
        customers = customers[:max_customers + 1]  # 含depot

    # depot = customers[0]
    depot_row = customers[0]
    cust_rows = customers[1:]

    coords  = np.array([[row[1], row[2]] for row in [depot_row] + cust_rows])
    demands = np.array([row[3] for row in [depot_row] + cust_rows])

    # EUC_2D 距离矩阵
    n       = len(coords)
    diff    = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.floor(np.sqrt((diff**2).sum(-1)) + 0.5).astype(np.int32)

    print(f"   ✅ 解析完成: {len(cust_rows)} 个客户, 容量={capacity}")
    return CVRPInstance(name, coords, demands, capacity, dist_mat)


# =============================================================================
# 模块三：CVRP 工具函数
# =============================================================================

def greedy_initial_solution(inst: CVRPInstance, rng: np.random.RandomState) -> list:
    """
    最近邻贪心构造初始解（每辆车依次从depot出发，贪心插入最近可行客户）。
    """
    unvisited = list(range(1, inst.n_customers + 1))
    rng.shuffle(unvisited)
    solution  = []

    while unvisited:
        route    = []
        load     = 0
        current  = 0   # depot
        while unvisited:
            # 找最近的、容量允许的客户
            best_c, best_d = None, float("inf")
            for c in unvisited:
                if load + inst.demands[c] <= inst.capacity:
                    d = inst.dist_matrix[current][c]
                    if d < best_d:
                        best_d, best_c = d, c
            if best_c is None:
                break   # 当前路线已满
            route.append(best_c)
            unvisited.remove(best_c)
            load    += inst.demands[best_c]
            current  = best_c
        if route:
            solution.append(route)

    return solution


def flatten_solution(solution: list) -> list:
    """展开多路线解为节点列表（不含depot）"""
    return [c for route in solution for c in route]


def solution_to_removal_list(solution: list) -> list:
    """返回所有客户节点（按路线展开）"""
    return flatten_solution(solution)


# =============================================================================
# 模块四：12 Destroy + 2 Repair 算子（CVRP版，对标论文）
# =============================================================================

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _removal_cost(inst: CVRPInstance, route: list, idx: int) -> float:
    """移除路线中idx位置节点后节省的距离（detour cost）"""
    d   = inst.dist_matrix
    n   = len(route)
    if n == 0:
        return 0.0
    prev = 0 if idx == 0 else route[idx - 1]
    curr = route[idx]
    nxt  = 0 if idx == n - 1 else route[idx + 1]
    return (d[prev][curr] + d[curr][nxt] - d[prev][nxt])


def _find_node_in_solution(solution: list, node: int):
    """返回 (route_idx, pos_in_route)"""
    for ri, route in enumerate(solution):
        for pi, c in enumerate(route):
            if c == node:
                return ri, pi
    return None, None


def _remove_nodes_from_solution(solution: list, nodes_to_remove: set) -> list:
    """从多路线解中移除一组节点，清除空路线"""
    new_sol = []
    for route in solution:
        new_r = [c for c in route if c not in nodes_to_remove]
        if new_r:
            new_sol.append(new_r)
    return new_sol


# ── 12 个 Destroy 算子 ────────────────────────────────────────────────────────

def destroy_random_node(inst: CVRPInstance, solution: list, d: int,
                        rng: random.Random, **kwargs):
    """1. Random Node Destroy: 随机移除d个节点"""
    all_nodes = flatten_solution(solution)
    removed   = rng.sample(all_nodes, min(d, len(all_nodes)))
    new_sol   = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_random_route(inst: CVRPInstance, solution: list, d: int,
                         rng: random.Random, **kwargs):
    """2. Random Route Destroy: 随机移除整条路线（或直到移除≥d个节点）"""
    if not solution:
        return solution, []
    removed  = []
    sol_copy = [r.copy() for r in solution]
    rng.shuffle(sol_copy)
    for route in sol_copy:
        removed.extend(route)
        if len(removed) >= d:
            break
    removed   = removed[:d]
    new_sol   = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_worst_node(inst: CVRPInstance, solution: list, d: int,
                       rng: random.Random, **kwargs):
    """3. Worst-Node Removal: 依次移除detour cost最高的节点"""
    sol_copy = [r.copy() for r in solution]
    removed  = []
    for _ in range(min(d, sum(len(r) for r in sol_copy))):
        best_cost, best_node, best_ri, best_pi = -1, None, -1, -1
        for ri, route in enumerate(sol_copy):
            for pi in range(len(route)):
                cost = _removal_cost(inst, route, pi)
                if cost > best_cost:
                    best_cost = cost
                    best_node = route[pi]
                    best_ri, best_pi = ri, pi
        if best_node is None:
            break
        removed.append(sol_copy[best_ri].pop(best_pi))
        sol_copy = [r for r in sol_copy if r]
    new_sol = [r for r in sol_copy if r]
    return new_sol, removed


def destroy_neighbourhood(inst: CVRPInstance, solution: list, d: int,
                          rng: random.Random, **kwargs):
    """4. Neighbourhood Removal: 以随机节点为中心，移除d个最近邻节点"""
    all_nodes = flatten_solution(solution)
    if not all_nodes:
        return solution, []
    seed_node = rng.choice(all_nodes)
    dists     = [(inst.dist_matrix[seed_node][c], c)
                 for c in all_nodes if c != seed_node]
    dists.sort()
    removed   = [c for _, c in dists[:d]]
    new_sol   = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_greedy_route(inst: CVRPInstance, solution: list, d: int,
                         rng: random.Random, **kwargs):
    """5. Greedy Route Destroy: 移除单位长度代价最高路线中的节点"""
    if not solution:
        return solution, []
    scored = sorted(solution,
                    key=lambda r: inst.route_cost(r) / (len(r) + EPSILON),
                    reverse=True)
    removed = []
    for route in scored:
        removed.extend(route)
        if len(removed) >= d:
            break
    removed = removed[:d]
    new_sol = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_proximity(inst: CVRPInstance, solution: list, d: int,
                      rng: random.Random, **kwargs):
    """6. Proximity Destroy (Shaw's): 按relatedness移除相似节点"""
    all_nodes = flatten_solution(solution)
    if not all_nodes:
        return solution, []
    removed   = [rng.choice(all_nodes)]
    candidates = [c for c in all_nodes if c != removed[0]]
    while len(removed) < d and candidates:
        # relatedness = 距离最近的已移除节点
        scored = sorted(candidates,
                        key=lambda c: min(inst.dist_matrix[c][r]
                                          for r in removed))
        # 带随机扰动的选择（不纯贪心）
        idx = int(rng.random() ** 3 * len(scored))
        removed.append(scored[idx])
        candidates.remove(scored[idx])
    new_sol = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_cluster(inst: CVRPInstance, solution: list, d: int,
                    rng: random.Random, **kwargs):
    """7. Cluster Destroy: 移除地理上最密集的d个节点"""
    all_nodes = flatten_solution(solution)
    if not all_nodes:
        return solution, []
    seed      = rng.choice(all_nodes)
    scored    = sorted(all_nodes,
                       key=lambda c: inst.dist_matrix[seed][c])
    removed   = scored[:d]
    new_sol   = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_node_neighbourhood(inst: CVRPInstance, solution: list, d: int,
                                rng: random.Random, **kwargs):
    """8. Node Neighbourhood Destroy: 基于路线结构的邻域移除"""
    all_nodes = flatten_solution(solution)
    if not all_nodes:
        return solution, []
    seed_node      = rng.choice(all_nodes)
    seed_ri, seed_pi = _find_node_in_solution(solution, seed_node)
    if seed_ri is None:
        return destroy_neighbourhood(inst, solution, d, rng)
    # 移除同路线中seed_node周围的节点，再按距离扩展
    route    = solution[seed_ri]
    route_nodes = set(route)
    scored   = sorted(all_nodes,
                      key=lambda c: (0 if c in route_nodes else 1,
                                     inst.dist_matrix[seed_node][c]))
    removed  = scored[:d]
    new_sol  = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_zone(inst: CVRPInstance, solution: list, d: int,
                 rng: random.Random, **kwargs):
    """9. Zone Destroy: 将坐标空间划分为网格，移除某个zone中的节点"""
    all_nodes = flatten_solution(solution)
    if not all_nodes:
        return solution, []
    coords  = inst.coords
    # 随机选择一个zone中心
    cx, cy  = coords[rng.choice(all_nodes)]
    zone_r  = max(coords[:, 0].max() - coords[:, 0].min(),
                  coords[:, 1].max() - coords[:, 1].min()) * 0.3
    in_zone = [c for c in all_nodes
               if abs(coords[c][0] - cx) <= zone_r
               and abs(coords[c][1] - cy) <= zone_r]
    if len(in_zone) < d:
        in_zone = sorted(all_nodes,
                          key=lambda c: (coords[c][0] - cx)**2
                                       + (coords[c][1] - cy)**2)
    removed = in_zone[:d]
    new_sol = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_route_neighbourhood(inst: CVRPInstance, solution: list, d: int,
                                 rng: random.Random, **kwargs):
    """10. Route Neighbourhood Destroy: 移除地理上相近的路线中的节点"""
    if not solution:
        return solution, []
    seed_route = rng.choice(solution)
    if not seed_route:
        return solution, []
    # 计算路线中心
    def route_center(r):
        return inst.coords[r].mean(axis=0)
    seed_center = route_center(seed_route)
    scored = sorted(solution,
                    key=lambda r: np.linalg.norm(route_center(r) - seed_center))
    removed = []
    for route in scored:
        removed.extend(route)
        if len(removed) >= d:
            break
    removed = removed[:d]
    new_sol = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_pair(inst: CVRPInstance, solution: list, d: int,
                 rng: random.Random, **kwargs):
    """11. Pair Destroy: 移除在多条路线中距离接近的节点对"""
    all_nodes = flatten_solution(solution)
    if len(all_nodes) < 2:
        return solution, all_nodes.copy()
    seed      = rng.choice(all_nodes)
    pairs     = sorted(all_nodes,
                       key=lambda c: inst.dist_matrix[seed][c])
    removed   = pairs[:d]
    new_sol   = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


def destroy_historical_pair(inst: CVRPInstance, solution: list, d: int,
                             rng: random.Random,
                             history_matrix: np.ndarray = None, **kwargs):
    """12. Historical Node-Pair Removal: 利用历史共现频率矩阵"""
    all_nodes = flatten_solution(solution)
    if not all_nodes:
        return solution, []
    if history_matrix is None:
        # 无历史信息时降级为proximity
        return destroy_proximity(inst, solution, d, rng)
    seed    = rng.choice(all_nodes)
    scored  = sorted([c for c in all_nodes if c != seed],
                     key=lambda c: -history_matrix[seed][c])
    removed = [seed] + scored[:d - 1]
    new_sol = _remove_nodes_from_solution(solution, set(removed))
    return new_sol, removed


# ── 2 个 Repair 算子 ──────────────────────────────────────────────────────────

def _cheapest_insertion_cost(inst: CVRPInstance, solution: list, node: int):
    """
    找到将node插入solution中代价最小的位置。
    返回 (min_cost, best_route_idx, best_pos, delta_list_for_regret)
    """
    d       = inst.dist_matrix
    best_cost = float("inf")
    best_ri, best_pos = -1, -1
    all_route_costs   = []   # 每条路线最优插入代价（用于k-regret）

    for ri, route in enumerate(solution):
        # 容量检查
        if inst.route_load(route) + inst.demands[node] > inst.capacity:
            all_route_costs.append(float("inf"))
            continue
        # 枚举插入位置
        min_cost_r, best_pos_r = float("inf"), -1
        for pos in range(len(route) + 1):
            prev = 0 if pos == 0 else route[pos - 1]
            nxt  = 0 if pos == len(route) else route[pos]
            cost = d[prev][node] + d[node][nxt] - d[prev][nxt]
            if cost < min_cost_r:
                min_cost_r = cost
                best_pos_r = pos
        all_route_costs.append(min_cost_r)
        if min_cost_r < best_cost:
            best_cost = min_cost_r
            best_ri, best_pos = ri, best_pos_r

    # 也考虑新建路线
    new_route_cost = d[0][node] + d[node][0]
    all_route_costs.append(new_route_cost)
    if new_route_cost < best_cost:
        best_cost = new_route_cost
        best_ri   = len(solution)   # 标记新建路线
        best_pos  = 0

    return best_cost, best_ri, best_pos, sorted(all_route_costs)


def repair_greedy(inst: CVRPInstance, solution: list, removed: list,
                  rng: random.Random = None) -> list:
    """Greedy Repair: 依次将每个节点插入代价最小的位置"""
    sol  = [r.copy() for r in solution]
    # 随机打乱插入顺序避免偏差
    nodes = removed.copy()
    if rng:
        rng.shuffle(nodes)
    for node in nodes:
        cost, ri, pos, _ = _cheapest_insertion_cost(inst, sol, node)
        if ri == len(sol):  # 新建路线
            sol.append([node])
        else:
            sol[ri].insert(pos, node)
    return [r for r in sol if r]


def repair_two_regret(inst: CVRPInstance, solution: list, removed: list,
                      rng: random.Random = None) -> list:
    """
    2-Regret Repair: 每次插入具有最大2-regret值的节点。
    regret(node) = 第2便宜插入代价 - 最便宜插入代价。
    """
    sol   = [r.copy() for r in solution]
    nodes = removed.copy()

    while nodes:
        best_regret, best_node = -float("inf"), None
        best_ri, best_pos      = -1, -1
        for node in nodes:
            cost, ri, pos, sorted_costs = _cheapest_insertion_cost(
                inst, sol, node
            )
            regret = (sorted_costs[1] - sorted_costs[0]
                      if len(sorted_costs) >= 2 else 0.0)
            if regret > best_regret:
                best_regret = regret
                best_node   = node
                best_ri, best_pos = ri, pos
        if best_node is None:
            break
        if best_ri == len(sol):
            sol.append([best_node])
        else:
            sol[best_ri].insert(best_pos, best_node)
        nodes.remove(best_node)

    return [r for r in sol if r]


# ── 算子注册表 ────────────────────────────────────────────────────────────────
DESTROY_REGISTRY = {
    "random_node":         destroy_random_node,
    "random_route":        destroy_random_route,
    "worst_node":          destroy_worst_node,
    "neighbourhood":       destroy_neighbourhood,
    "greedy_route":        destroy_greedy_route,
    "proximity":           destroy_proximity,
    "cluster":             destroy_cluster,
    "node_neighbourhood":  destroy_node_neighbourhood,
    "zone":                destroy_zone,
    "route_neighbourhood": destroy_route_neighbourhood,
    "pair":                destroy_pair,
    "historical_pair":     destroy_historical_pair,
}

REPAIR_REGISTRY = {
    "greedy":     repair_greedy,
    "two_regret": repair_two_regret,
}

EPSILON = 1e-8


# =============================================================================
# 模块五：MDP 独立评估框架（对标论文 Table 1：累积奖励）
# =============================================================================

class MDPEvaluator:
    """
    论文中的 MDP 独立评估框架。
    给定一个初始解和 operator budget，
    让 agent 反复执行 (destroy, repair) 对，
    最终返回累积奖励 = F(S0) - F(S_best)。
    """
    def __init__(self, inst: CVRPInstance,
                 destroy_ops: list, repair_ops: list,
                 budget: int, destroy_scale: int):
        self.inst         = inst
        self.destroy_ops  = destroy_ops
        self.repair_ops   = repair_ops
        self.budget       = budget
        self.d            = destroy_scale

    def run_episode(self, initial_solution: list,
                    policy_fn,            # fn(solution, stats) -> (d_op, r_op)
                    rng,
                    history_matrix=None) -> float:
        """
        执行一个 MDP episode。
        policy_fn: 接收 (current_solution, op_stats) -> (destroy_op_name, repair_op_name)
        返回: cumulative_reward = F(S0) - F(S_best)
        """
        inst     = self.inst
        solution = [r.copy() for r in initial_solution]
        f0       = inst.solution_cost(initial_solution)
        f_best   = f0

        op_stats = {op: {"used": 0, "improved": 0, "score": 0.0}
                    for op in self.destroy_ops + self.repair_ops}

        for _ in range(self.budget):
            # 选算子
            d_op, r_op = policy_fn(solution, op_stats, rng)

            # Destroy
            d_fn   = DESTROY_REGISTRY[d_op]
            new_sol, removed = d_fn(
                inst, solution, self.d, rng,
                history_matrix=history_matrix
            )

            # Repair
            r_fn    = REPAIR_REGISTRY[r_op]
            new_sol = r_fn(inst, new_sol, removed, rng)

            new_cost = inst.solution_cost(new_sol)
            op_stats[d_op]["used"] += 1
            op_stats[r_op]["used"] += 1

            if new_cost < f_best:
                f_best  = new_cost
                op_stats[d_op]["improved"] += 1
                op_stats[r_op]["improved"] += 1
                op_stats[d_op]["score"]    += 1.0
                op_stats[r_op]["score"]    += 1.0
                solution = new_sol
            elif new_cost < inst.solution_cost(solution):
                solution = new_sol

        return f0 - f_best   # cumulative reward


def policy_random(solution, op_stats, rng, destroy_ops, repair_ops):
    """RAN: 均匀随机选算子"""
    return rng.choice(destroy_ops), rng.choice(repair_ops)


def make_lrw_policy(destroy_ops: list, repair_ops: list,
                    rho: float = 0.5, min_prob: float = 0.1):
    """
    LRW (Learned Roulette Wheel): 用 Laplace 平滑的轮盘赌。
    在 MDP 内部根据 op_stats 动态更新权重。
    """
    d_weights = {op: 1.0 for op in destroy_ops}
    r_weights = {op: 1.0 for op in repair_ops}

    def policy(solution, op_stats, rng):
        # 更新权重
        for op in destroy_ops:
            used = op_stats[op]["used"]
            if used > 0:
                avg_s = op_stats[op]["score"] / used
                d_weights[op] = (1 - rho) * d_weights[op] + rho * avg_s
        for op in repair_ops:
            used = op_stats[op]["used"]
            if used > 0:
                avg_s = op_stats[op]["score"] / used
                r_weights[op] = (1 - rho) * r_weights[op] + rho * avg_s

        # 轮盘赌选算子
        def roulette(weights_dict):
            total = sum(weights_dict.values())
            if total <= 0:
                return rng.choice(list(weights_dict.keys()))
            pick, cum = rng.random() * total, 0.0
            for name, w in weights_dict.items():
                cum += w
                if pick <= cum:
                    return name
            return rng.choice(list(weights_dict.keys()))

        return roulette(d_weights), roulette(r_weights)

    return policy


def make_llm_mdp_policy(destroy_ops: list, repair_ops: list,
                         inst: CVRPInstance,
                         ask_llm_fn):
    """
    SC-LLM-OS MDP 策略: 根据 LLM 决策调整算子权重。
    """
    weights = {op: 1.0 / len(destroy_ops) for op in destroy_ops}
    call_count = [0]
    TRIGGER_EVERY = 3   # 每3步触发一次LLM（MDP内budget=10，触发3次）

    def policy(solution, op_stats, rng):
        call_count[0] += 1

        if call_count[0] % TRIGGER_EVERY == 0:
            # 提取MDP内状态
            meta = _mdp_state_to_json(solution, inst, op_stats, destroy_ops)
            decision = ask_llm_fn(meta)
            mode = decision.get("mode", "explore_global")
            _update_weights_from_mode(weights, destroy_ops, mode, op_stats)

        # 加权轮盘赌选 destroy
        d_op = _roulette_select(weights, rng)
        # 加权轮盘赌选 repair（简化：平均权重）
        r_op = rng.choice(repair_ops)
        return d_op, r_op

    return policy


def _roulette_select(weights: dict, rng) -> str:
    total = sum(weights.values()) or 1.0
    pick, cum = rng.random() * total, 0.0
    for name, w in weights.items():
        cum += w
        if pick <= cum:
            return name
    return rng.choice(list(weights.keys()))


def _update_weights_from_mode(weights: dict, destroy_ops: list,
                               mode: str, op_stats: dict):
    """根据LLM模式和算子历史综合更新权重"""
    # 基于历史表现的自适应分量
    for op in destroy_ops:
        used = op_stats.get(op, {}).get("used", 1)
        score = op_stats.get(op, {}).get("score", 1.0)
        weights[op] = max(score / (used + EPSILON), 0.01)

    # LLM偏置分量
    mode_boost = {
        "exploit": {
            "worst_node": 1.5, "greedy_route": 1.3, "neighbourhood": 1.2
        },
        "explore_topology": {
            "proximity": 1.5, "cluster": 1.3, "zone": 1.2,
            "route_neighbourhood": 1.2
        },
        "explore_global": {
            "random_node": 1.5, "random_route": 1.3, "historical_pair": 1.2
        },
    }
    boost = mode_boost.get(mode, {})
    for op in destroy_ops:
        weights[op] *= boost.get(op, 1.0)

    # 归一化
    total = sum(weights.values()) or 1.0
    for op in destroy_ops:
        weights[op] = max(weights[op] / total, MIN_PROB)
    total2 = sum(weights.values())
    for op in destroy_ops:
        weights[op] /= total2


def _mdp_state_to_json(solution, inst, op_stats, destroy_ops) -> str:
    cost     = inst.solution_cost(solution)
    n_routes = len(solution)
    fmt_ops  = {}
    for op in destroy_ops:
        st = op_stats.get(op, {"used": 0, "score": 0.0, "improved": 0})
        used = st["used"]
        fmt_ops[op] = {
            "used":         used,
            "success_rate": round(st["improved"] / used, 3) if used > 0 else 0.0,
            "avg_score":    round(st["score"] / used, 3)    if used > 0 else 0.0,
        }
    meta = {
        "current_cost":    round(float(cost), 2),
        "n_routes":        n_routes,
        "operator_feedback": fmt_ops,
    }
    return json.dumps(meta, indent=2, ensure_ascii=False)


# =============================================================================
# 模块六：LLM 大脑（CVRP版，与原始保持一致）
# =============================================================================

def ask_llm_for_mode(meta_state_json: str) -> dict:
    """调用 LLM API，返回宏观战术模式。与 v7.0 逻辑一致。"""
    prompt = f"""你是一个CVRP组合优化算法的宏观战略指挥官。
你的任务是根据当前搜索状态，选择最合适的算子选择策略模式。

【当前系统状态 JSON】
{meta_state_json}

【战术模式定义】
- exploit          : 精细开发，偏置 worst_node/greedy_route/neighbourhood 算子
- explore_topology : 拓扑探索，偏置 proximity/cluster/zone/route_neighbourhood 算子
- explore_global   : 全局扰动，偏置 random_node/random_route/historical_pair 算子

【决策规则】
1. 若某算子 avg_score > 0.5，优先选择与其匹配的模式；
2. 若所有算子 avg_score < 0.2 且used > 3，选 explore_global；
3. 若 n_routes 偏多（路线数 > 理论最优的1.3倍），选 explore_topology。

严格返回以下JSON，不输出其他内容：
{{
    "reasoning": "一句话说明选择该模式的逻辑",
    "mode": "<exploit|explore_topology|explore_global>"
}}
"""
    max_retry = 3
    for attempt in range(1, max_retry + 1):
        try:
            time.sleep(2)
            if LLM_PROVIDER == "gemini":
                from google import genai
                client = genai.Client(api_key=API_KEY)
                res    = client.models.generate_content(
                    model    = "gemini-2.5-pro-preview-05-06",
                    contents = prompt,
                    config   = {"response_mime_type": "application/json",
                                "temperature": 0.2},
                )
                result = json.loads(res.text)
            elif LLM_PROVIDER == "deepseek":
                url     = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
                headers = {"Content-Type": "application/json",
                           "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
                data = json.dumps({
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2
                }).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    content = json.loads(
                        resp.read().decode("utf-8")
                    )["choices"][0]["message"]["content"]
                    result  = json.loads(content)
            elif LLM_PROVIDER == "cuhk":
                from openai import OpenAI
                import httpx
                # Disable proxies explicitly to prevent system proxy from routing internal requests to external proxy
                client = OpenAI(
                    api_key=CUHK_API_KEY,
                    base_url="https://ai.cuhk.edu.cn/open/v1",
                    timeout=httpx.Timeout(120.0),
                    http_client=httpx.Client(),
                )
                res = client.chat.completions.create(
                    model=CUHK_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. You must output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = res.choices[0].message.content
                result = json.loads(content)
            else:
                raise ValueError(f"未知LLM: {LLM_PROVIDER}")

            if result.get("mode") not in (
                "exploit", "explore_topology", "explore_global"
            ):
                raise ValueError(f"非法mode: {result.get('mode')}")
            result["is_fallback"] = False
            return result

        except Exception as e:
            print(f"   ⚠️  API失败 (attempt {attempt}): {e}")
            time.sleep(5 * attempt)

    return {"reasoning": "Fallback", "mode": "explore_global",
            "is_fallback": True}


# =============================================================================
# 模块七：Portfolio 规模扫描（核心评估入口，对标论文 Table 1/2）
# =============================================================================

def run_portfolio_sweep(
    inst_small: CVRPInstance,    # 小规模实例 (n=20) 用于MDP评估
    inst_full:  CVRPInstance,    # 完整实例 (n=100) 用于ALNS评估
    seeds:      list,
    n_init:     int,
    portfolio_sizes: list,
    destroy_scale:   int,
    budget:          int,
    mode:            str,        # "mdp" or "alns"
) -> dict:
    """
    对不同 Portfolio 规模 (|D|=2..12) 进行扫描实验。
    返回格式:
    {
        strategy: {
            portfolio_size: {
                "mean_reward" or "avg_obj": float,
                "best_obj": float,
                "values": [...]
            }
        }
    }
    """
    results = {s: {} for s in STRATEGIES}
    inst    = inst_small if mode == "mdp" else inst_full

    for n_d in portfolio_sizes:
        destroy_ops = DESTROY_OPS_ALL[:n_d]
        repair_ops  = REPAIR_OPS_ALL   # 始终使用 greedy + 2-regret

        print(f"\n  [Portfolio |D|={n_d}] mode={mode}, "
              f"ops={destroy_ops[:3]}{'...' if n_d>3 else ''}")

        for strategy in STRATEGIES:
            values = []
            for seed in seeds:
                rng_init = np.random.RandomState(seed)
                rng      = random.Random(seed)

                # 生成初始解集合（每个seed用128个初始解的均值评估）
                episode_rewards = []
                for _ in range(min(n_init, 16)):   # 调试时用16，正式用128
                    init_sol = greedy_initial_solution(inst, rng_init)

                    if mode == "mdp":
                        evaluator = MDPEvaluator(
                            inst, destroy_ops, repair_ops,
                            budget, destroy_scale
                        )
                        if strategy == "random":
                            policy = lambda sol, stats, r, d=destroy_ops, rep=repair_ops: \
                                policy_random(sol, stats, r, d, rep)
                        elif strategy == "lrw":
                            policy = make_lrw_policy(destroy_ops, repair_ops)
                        else:  # sc_llm_os
                            policy = make_llm_mdp_policy(
                                destroy_ops, repair_ops, inst, ask_llm_for_mode
                            )
                        reward = evaluator.run_episode(init_sol, policy, rng)
                        episode_rewards.append(reward)
                    else:
                        # ALNS 集成评估
                        res = run_alns(
                            inst, init_sol, destroy_ops, repair_ops,
                            strategy, seed, destroy_scale
                        )
                        episode_rewards.append(res["best_cost"])

                values.append(float(np.mean(episode_rewards)))

            results[strategy][n_d] = {
                "mean": round(float(np.mean(values)), 2),
                "std":  round(float(np.std(values)),  2),
                "values": values,
            }
            if mode == "alns":
                results[strategy][n_d]["best"] = round(float(np.min(values)), 2)
            print(f"    [{strategy:<12}] mean={'mean_reward' if mode=='mdp' else 'avg_obj'}"
                  f"={results[strategy][n_d]['mean']:.2f} "
                  f"± {results[strategy][n_d]['std']:.2f}")

    return results


# =============================================================================
# 模块八：ALNS 集成求解器（CVRP版）
# =============================================================================

def run_alns(
    inst:        CVRPInstance,
    initial_sol: list,
    destroy_ops: list,
    repair_ops:  list,
    strategy:    str,
    solver_seed: int,
    destroy_scale: int = 4,
    n_iter:      int   = None,
) -> dict:
    """
    CVRP版 ALNS 核心求解器。
    策略: random (RAN) / lrw (LRW) / sc_llm_os (本文方法)
    """
    rng = random.Random(solver_seed)
    np.random.seed(solver_seed)

    num_customers = inst.n_customers
    if n_iter is None:
        n_iter = max(300, num_customers * 3)

    # 初始化
    current_sol  = [r.copy() for r in initial_sol]
    best_sol     = [r.copy() for r in initial_sol]
    best_cost    = inst.solution_cost(best_sol)
    current_cost = best_cost

    # 算子权重（含Laplace伪计数）
    d_weights = {op: 1.0 for op in destroy_ops}
    r_weights = {op: 1.0 for op in repair_ops}

    op_stats = {op: {"used": 1, "improved": 0, "score": 1.0}
                for op in destroy_ops + repair_ops}

    # 历史共现矩阵（用于 historical_pair 算子）
    history_matrix = np.zeros(
        (num_customers + 1, num_customers + 1), dtype=np.float32
    )

    distance_history = [best_cost]
    llm_log          = []
    stagnation       = 0
    STAGNATION_THRESH = max(30, num_customers // 3)
    SEGMENT_LEN       = max(20, num_customers // 5)
    last_llm_iter     = 0

    for it in range(1, n_iter + 1):
        # SA温度
        progress    = it / n_iter
        temperature = best_cost * 0.05 * (0.001 ** progress)

        # 阶段性权重更新
        if it % SEGMENT_LEN == 0:
            if strategy == "random":
                for op in destroy_ops:
                    d_weights[op] = 1.0 / len(destroy_ops)
                for op in repair_ops:
                    r_weights[op] = 1.0 / len(repair_ops)

            elif strategy == "lrw":
                for op in destroy_ops:
                    used = op_stats[op]["used"]
                    avg  = op_stats[op]["score"] / used
                    d_weights[op] = max(
                        (1 - ALNS_RHO) * d_weights[op] + ALNS_RHO * avg,
                        MIN_PROB
                    )
                for op in repair_ops:
                    used = op_stats[op]["used"]
                    avg  = op_stats[op]["score"] / used
                    r_weights[op] = max(
                        (1 - ALNS_RHO) * r_weights[op] + ALNS_RHO * avg,
                        MIN_PROB
                    )
                # 归一化
                dt = sum(d_weights.values()); rt = sum(r_weights.values())
                d_weights = {k: v/dt for k,v in d_weights.items()}
                r_weights = {k: v/rt for k,v in r_weights.items()}

            elif strategy == "sc_llm_os":
                emergency = (stagnation >= STAGNATION_THRESH
                             and (it - last_llm_iter) >= 20)
                if emergency:
                    last_llm_iter = it
                    meta     = _mdp_state_to_json(
                        current_sol, inst, op_stats, destroy_ops
                    )
                    decision = ask_llm_for_mode(meta)
                    mode     = decision.get("mode", "explore_global")
                    _update_weights_from_mode(d_weights, destroy_ops,
                                              mode, op_stats)
                    for op in repair_ops:
                        r_weights[op] = 1.0 / len(repair_ops)
                    print(f"   [🧠 iter={it}] {mode} | "
                          f"stagnation={stagnation}")
                    llm_log.append({
                        "iteration": it, "mode": mode,
                        "stagnation": stagnation,
                        "best_cost": round(best_cost, 2),
                        "is_fallback": decision.get("is_fallback", False),
                    })

            # 重置阶段统计（保留伪计数）
            op_stats = {op: {"used": 1, "improved": 0, "score": 1.0}
                        for op in destroy_ops + repair_ops}

        # ── 算子选择与执行 ──────────────────────────────────────────────────
        d_op = _roulette_select(d_weights, rng)
        r_op = _roulette_select(r_weights, rng)

        d_fn   = DESTROY_REGISTRY[d_op]
        new_sol, removed = d_fn(
            inst, current_sol, destroy_scale, rng,
            history_matrix=history_matrix
        )
        r_fn    = REPAIR_REGISTRY[r_op]
        new_sol = r_fn(inst, new_sol, removed, rng)
        new_cost = inst.solution_cost(new_sol)

        # 更新历史矩阵（同一路线的节点共现）
        for route in new_sol:
            for i, a in enumerate(route):
                for b in route[i+1:]:
                    history_matrix[a][b] += 1
                    history_matrix[b][a] += 1

        # ── 接受准则 ───────────────────────────────────────────────────────
        delta = new_cost - current_cost
        op_stats[d_op]["used"] += 1
        op_stats[r_op]["used"] += 1

        if new_cost < best_cost:
            best_cost   = new_cost
            best_sol    = [r.copy() for r in new_sol]
            current_sol = [r.copy() for r in new_sol]
            current_cost = new_cost
            op_stats[d_op]["score"]    += 3.0
            op_stats[d_op]["improved"] += 1
            op_stats[r_op]["score"]    += 3.0
            op_stats[r_op]["improved"] += 1
            stagnation = 0
        elif delta < 0:
            current_sol  = [r.copy() for r in new_sol]
            current_cost = new_cost
            op_stats[d_op]["score"] += 1.5
            op_stats[r_op]["score"] += 1.5
            stagnation = max(0, stagnation - 1)
        else:
            accept_p = math.exp(-delta / max(temperature, 1e-6))
            if rng.random() < accept_p:
                current_sol  = [r.copy() for r in new_sol]
                current_cost = new_cost
            stagnation += 1

        distance_history.append(best_cost)

    return {
        "best_cost":        best_cost,
        "best_solution":    best_sol,
        "distance_history": distance_history,
        "llm_log":          llm_log,
    }


# =============================================================================
# 模块九：Destroy Scale 扫描（对标论文 Figure 2 下半部分）
# =============================================================================

def run_scale_sweep(
    inst:         CVRPInstance,
    seeds:        list,
    n_init:       int,
    destroy_scales: list,
    portfolio_size: int,    # 固定使用最大Portfolio |D|=12
    budget:         int,
) -> dict:
    """
    固定 |D|=portfolio_size，变化 destroy scale d∈destroy_scales。
    在 MDP 框架下评估累积奖励。
    """
    destroy_ops = DESTROY_OPS_ALL[:portfolio_size]
    repair_ops  = REPAIR_OPS_ALL
    results     = {s: {} for s in STRATEGIES}

    print(f"\n{'='*50}")
    print(f"  Destroy Scale Sweep | |D|={portfolio_size}")
    print(f"{'='*50}")

    for d_scale in destroy_scales:
        print(f"\n  [d={d_scale}]")
        for strategy in STRATEGIES:
            values = []
            for seed in seeds:
                rng_init = np.random.RandomState(seed)
                rng      = random.Random(seed)
                rewards  = []
                for _ in range(min(n_init, 16)):
                    init_sol  = greedy_initial_solution(inst, rng_init)
                    evaluator = MDPEvaluator(
                        inst, destroy_ops, repair_ops, budget, d_scale
                    )
                    if strategy == "random":
                        policy = lambda sol, stats, r, d=destroy_ops, rep=repair_ops: \
                            policy_random(sol, stats, r, d, rep)
                    elif strategy == "lrw":
                        policy = make_lrw_policy(destroy_ops, repair_ops)
                    else:
                        policy = make_llm_mdp_policy(
                            destroy_ops, repair_ops, inst, ask_llm_for_mode
                        )
                    rewards.append(evaluator.run_episode(init_sol, policy, rng))
                values.append(float(np.mean(rewards)))

            results[strategy][d_scale] = {
                "mean": round(float(np.mean(values)), 2),
                "std":  round(float(np.std(values)),  2),
            }
            print(f"    [{strategy:<12}] d={d_scale} "
                  f"mean_reward={results[strategy][d_scale]['mean']:.2f}")

    return results


# =============================================================================
# 模块十：可视化（对标论文图表）
# =============================================================================

def plot_portfolio_sweep(sweep_results: dict, instance_name: str,
                          mode: str, output_dir: str):
    """
    绘制 Portfolio 规模扫描结果（对标论文 Table 1/2 可视化版）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    style_map = {
        "random":    ("gray",       "--", "RAN (Random)"),
        "lrw":       ("darkorange", "-.", "LRW (Learned RW)"),
        "sc_llm_os": ("royalblue",  "-",  "SC-LLM-OS (Ours)"),
    }
    x_sizes = sorted(PORTFOLIO_SIZES)
    metric  = "mean" if mode == "mdp" else "mean"
    ylabel  = "Cumulative Reward (↑)" if mode == "mdp" else "Avg Objective Value (↓)"

    for strategy in STRATEGIES:
        color, ls, label = style_map[strategy]
        y_vals = [sweep_results[strategy].get(n, {}).get(metric, 0)
                  for n in x_sizes]
        y_stds = [sweep_results[strategy].get(n, {}).get("std", 0)
                  for n in x_sizes]
        y = np.array(y_vals)
        e = np.array(y_stds)
        ax.plot(x_sizes, y, color=color, linestyle=ls,
                linewidth=2.2, marker="o", markersize=5, label=label)
        ax.fill_between(x_sizes, y - e, y + e,
                        alpha=0.15, color=color)

    ax.set_xlabel("|D| (Destroy Portfolio Size)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"Portfolio Size Sweep | {instance_name} | mode={mode}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_sizes)
    plt.tight_layout()
    path = os.path.join(output_dir, f"portfolio_sweep_{instance_name}_{mode}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Portfolio扫描图已保存: {path}")


def plot_scale_sweep(scale_results: dict, instance_name: str, output_dir: str):
    """绘制 Destroy Scale 扫描结果（对标论文 Figure 2 下半）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    style_map = {
        "random":    ("gray",       "--", "RAN"),
        "lrw":       ("darkorange", "-.", "LRW"),
        "sc_llm_os": ("royalblue",  "-",  "SC-LLM-OS"),
    }
    for strategy in STRATEGIES:
        color, ls, label = style_map[strategy]
        scales = sorted(scale_results[strategy].keys())
        y = [scale_results[strategy][d]["mean"] for d in scales]
        e = [scale_results[strategy][d]["std"]  for d in scales]
        ax.plot(scales, y, color=color, linestyle=ls,
                linewidth=2.2, marker="s", markersize=5, label=label)
        ax.fill_between(scales,
                        np.array(y) - np.array(e),
                        np.array(y) + np.array(e),
                        alpha=0.15, color=color)
    ax.set_xlabel("Destroy Scale d", fontsize=11)
    ax.set_ylabel("Cumulative Reward (↑)", fontsize=11)
    ax.set_title(f"Destroy Scale Impact | {instance_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"scale_sweep_{instance_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Scale扫描图已保存: {path}")


def plot_alns_boxplot(alns_results: dict, instance_name: str, output_dir: str):
    """ALNS集成评估箱线图（对标论文 Table 2 可视化版）"""
    fig, ax = plt.subplots(figsize=(8, 5))
    data    = [alns_results[s] for s in STRATEGIES]
    
    labels_map = {"random": "RAN", "lrw": "LRW", "sc_llm_os": "SC-LLM-OS"}
    colors_map = {"random": "gray", "lrw": "darkorange", "sc_llm_os": "royalblue"}
    labels  = [labels_map[s] for s in STRATEGIES]
    colors  = [colors_map[s] for s in STRATEGIES]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, (s, c) in enumerate(zip(STRATEGIES, colors), 1):
        mv = float(np.mean(alns_results[s])) if alns_results[s] else 0
        ax.text(i, mv, f"  μ={mv:.1f}", fontsize=8, color=c,
                va="center", fontweight="bold")
    ax.set_ylabel("Best Objective Value (↓)", fontsize=11)
    ax.set_title(f"ALNS Integration Results | {instance_name}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"alns_boxplot_{instance_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 ALNS箱线图已保存: {path}")


# =============================================================================
# 模块十一：论文格式结果表打印
# =============================================================================

def print_paper_style_table(sweep_results: dict, instance_name: str, mode: str):
    """
    仿照论文 Table 1/Table 2 格式打印结果
    """
    metric = "mean"
    header = "Cumulative Reward (↑)" if mode == "mdp" else "Avg Obj (↓) / Best Obj (↓)"
    width  = 80

    print("\n" + "=" * width)
    print(f"  {instance_name}  |  Mode={mode.upper()}  |  {header}")
    print(f"  {'|D|':<5}", end="")
    for s in STRATEGIES:
        label = {"random": "RAN", "lrw": "LRW", "sc_llm_os": "SC-LLM-OS"}[s]
        print(f"  {label:>18}", end="")
    print()
    print("-" * width)

    x_sizes = sorted(PORTFOLIO_SIZES)
    for n_d in x_sizes:
        print(f"  {n_d:<5}", end="")
        for s in STRATEGIES:
            d = sweep_results[s].get(n_d, {})
            m = d.get("mean", 0.0)
            e = d.get("std",  0.0)
            print(f"  {m:>9.1f}±{e:<6.1f}", end="")
        print()

    # 均值行
    print("-" * width)
    print(f"  {'mean':<5}", end="")
    for s in STRATEGIES:
        vals = [sweep_results[s].get(n, {}).get("mean", 0.0) for n in x_sizes]
        m    = float(np.mean(vals))
        e    = float(np.std(vals))
        print(f"  {m:>9.1f}±{e:<6.1f}", end="")
    print()
    print("=" * width)


def save_sweep_results(results: dict, name: str, output_dir: str, run_id: str):
    path = os.path.join(output_dir, f"{name}_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   💾 结果已保存: {path}")


# =============================================================================
# 模块十二：主程序入口
# =============================================================================

if __name__ == "__main__":

    # ── API 校验 ─────────────────────────────────────────────────────────────
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise SystemExit("❌ DEEPSEEK_API_KEY 未设置！")
        print("✅ DeepSeek API Key 校验通过")
    elif LLM_PROVIDER == "gemini":
        if not API_KEY:
            raise SystemExit("❌ GEMINI_API_KEY 未设置！")
        print("✅ Gemini API Key 校验通过")
    elif LLM_PROVIDER == "cuhk":
        if not CUHK_API_KEY:
            raise SystemExit("❌ CUHK_API_KEY 未设置！")
        if not CUHK_MODEL_NAME:
            raise SystemExit("❌ CUHK_MODEL_NAME 未设置！")
        print("✅ CUHK OpenAI API 校验通过")

    print("\n" + "="*60)
    print(f"  SC-LLM-OS v8.0 | CVRP/Solomon Edition")
    print(f"  LLM Provider : {LLM_PROVIDER}")
    print(f"  Instances    : {SOLOMON_INSTANCES}")
    print(f"  Seeds        : {SEEDS}")
    print(f"  Portfolio    : |D|={PORTFOLIO_SIZES}")
    print(f"  Scales       : {DESTROY_SCALES}")
    print("="*60)

    for inst_name in SOLOMON_INSTANCES:

        # ── 加载数据 ─────────────────────────────────────────────────────────
        # 小规模：前20个客户（MDP训练/评估，与论文一致）
        inst_small = load_solomon(inst_name, max_customers=MDP_SMALL_N)
        # 完整规模：100个客户（ALNS集成评估）
        inst_full  = load_solomon(inst_name, max_customers=100)

        inst_dir = os.path.join(OUTPUT_DIR, inst_name)
        os.makedirs(inst_dir, exist_ok=True)

        # ══════════════════════════════════════════════════════════════════════
        # 实验一：MDP 独立评估 + Portfolio 规模扫描（对标论文 Table 1）
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  实验一: MDP Portfolio Sweep | {inst_name}")
        print(f"{'='*60}")

        mdp_sweep = run_portfolio_sweep(
            inst_small    = inst_small,
            inst_full     = inst_full,
            seeds         = SEEDS,
            n_init        = N_INIT_SOLUTIONS,
            portfolio_sizes = PORTFOLIO_SIZES,
            destroy_scale = MDP_DESTROY_SCALE,
            budget        = MDP_BUDGET,
            mode          = "mdp",
        )
        print_paper_style_table(mdp_sweep, inst_name, mode="mdp")
        save_sweep_results(mdp_sweep, f"mdp_sweep_{inst_name}", inst_dir, RUN_ID)
        plot_portfolio_sweep(mdp_sweep, inst_name, "mdp", inst_dir)

        # ══════════════════════════════════════════════════════════════════════
        # 实验二：ALNS 集成评估 + Portfolio 规模扫描（对标论文 Table 2）
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  实验二: ALNS Portfolio Sweep | {inst_name}")
        print(f"{'='*60}")

        alns_sweep = run_portfolio_sweep(
            inst_small    = inst_small,
            inst_full     = inst_full,
            seeds         = SEEDS,
            n_init        = N_INIT_SOLUTIONS,
            portfolio_sizes = PORTFOLIO_SIZES,
            destroy_scale = MDP_DESTROY_SCALE,
            budget        = MDP_BUDGET,
            mode          = "alns",
        )
        print_paper_style_table(alns_sweep, inst_name, mode="alns")
        save_sweep_results(alns_sweep, f"alns_sweep_{inst_name}", inst_dir, RUN_ID)
        plot_portfolio_sweep(alns_sweep, inst_name, "alns", inst_dir)

        # ALNS箱线图（最大portfolio |D|=12）
        alns_final = {s: alns_sweep[s].get(12, {}).get("values", [])
                      for s in STRATEGIES}
        plot_alns_boxplot(alns_final, inst_name, inst_dir)

        # ══════════════════════════════════════════════════════════════════════
        # 实验三：Destroy Scale 扫描（对标论文 Figure 2 下半）
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  实验三: Destroy Scale Sweep | {inst_name}")
        print(f"{'='*60}")

        scale_results = run_scale_sweep(
            inst          = inst_small,
            seeds         = SEEDS,
            n_init        = N_INIT_SOLUTIONS,
            destroy_scales = DESTROY_SCALES,
            portfolio_size = 12,
            budget         = MDP_BUDGET,
        )
        save_sweep_results(scale_results, f"scale_sweep_{inst_name}", inst_dir, RUN_ID)
        plot_scale_sweep(scale_results, inst_name, inst_dir)

    print("\n" + "="*60)
    print(f"  🎉 所有实验完成！结果保存于: {OUTPUT_DIR}")
    print("="*60)