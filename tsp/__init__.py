"""
SC-LLM-OS: Strategic Commander LLM for Optimization Scheduling
TSP 模块化版本

模块结构:
  config.py      - 全局配置 (API/实验参数/输出目录/缓存)
  data_loader.py - TSPLIB 数据加载
  utils.py       - 工具函数 (距离计算/轮盘赌/触发间隔)
  operators.py   - 算子库 (2-opt/破坏/修复)
  state.py       - S2N 状态提取
  llm_brain.py   - LLM 大脑 (API调用)
  solver.py      - 核心求解器
  runner.py      - 并行/串行执行器
  visualize.py   - 可视化
  stats.py       - 结果持久化与统计
  main.py        - 主程序入口
"""

from .config import (
    LLM_PROVIDER,
    API_KEY,
    DEEPSEEK_API_KEY,
    MIMO_API_KEY,
    INSTANCES,
    SEEDS,
    STRATEGIES,
    DESTRUCTION_RATIO,
    DESTROY_OPS,
    REPAIR_OPS,
    USE_TWO_OPT,
    USE_ELITE_RESTART,
    LLM_TRIGGER_ONLY_ON_STAGNATION,
    OUTPUT_DIR,
    RUN_ID,
    VERSION,
    load_cache,
    save_cache,
)
from .data_loader import load_tsplib_data
from .utils import calc_route_distance, select_by_roulette, calc_trigger_interval
from .operators import (
    OPERATORS,
    apply_two_opt,
    greedy_initial_route,
    destroy_random,
    destroy_worst,
    destroy_segment,
    destroy_shaw,
    destroy_long_edge,
    destroy_multi_segment,
    repair_greedy,
    repair_farthest,
    repair_random_order,
    repair_regret2,
)
from .state import state_to_meta_json
from .llm_brain import ask_llm_for_mode
from .solver import run_solver
from .runner import run_non_llm_parallel, run_llm_serial
from .visualize import plot_results, plot_llm_decision_timeline
from .stats import run_significance_tests, save_results, print_stats_table
