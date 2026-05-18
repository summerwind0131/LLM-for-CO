# =============================================================================
#  模块一：全局配置
#  - 控制台编码修复
#  - 环境变量 / API 配置
#  - 实验规模与算法超参数
#  - 输出目录
#  - matplotlib 中文字体自适应 [EXP-11]
#  - 缓存存取函数
# =============================================================================

import os
import sys
import json
import time
import platform

import matplotlib.pyplot as plt


# ── 控制台编码修复 ────────────────────────────────────────────────────────────

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── 环境变量 ──────────────────────────────────────────────────────────────────

# 尝试自动从 .env 文件加载环境变量（适用于服务器运行）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── API ───────────────────────────────────────────────────────────────────────
# LLM 供应商选择 (可选: "gemini" 或 "deepseek")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER").lower()

# Gemini 配置
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# DeepSeek 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

# Xiaomi Mimo 配置
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "").strip()
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.mimo.xiaomi.com/v1").strip()
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro").strip()


# ── 实验规模 ──────────────────────────────────────────────────────────────────
INSTANCES  = ["berlin52","kroA100","ch150","kroA200","ali535","d657","pr1002"]   # 多个规模用于泛化验证
SEEDS      = list(range(42, 72))  # [EXP-13] 30 个连续的整数，验证统计显著性
STRATEGIES = [
    "baseline",
    "traditional_alns",
    "sc_rule_os",
    "sc_random_os",
    "sc_fallback_os",
    "sc_llm_os",
]

# ── 算法超参数 ────────────────────────────────────────────────────────────────
# NUM_ITERATIONS 和 STAGNATION_RESET_THRESHOLD 将在各个实例加载后动态计算
# LLM_TRIGGER_INTERVAL 延后到加载数据时动态计算
DESTRUCTION_RATIO         = 0.06   # 动态破坏比例
USE_TWO_OPT               = False   # [EXP-1] 开启局部搜索
USE_ELITE_RESTART         = False  # [EXP-12] 精英解重启机制开关（当前停用）
LLM_TRIGGER_ONLY_ON_STAGNATION = False  # 仅在停滞(应急)时触发 LLM 策略调控，关闭固定周期触发
MAX_TWO_OPT_PASSES        = 2

# ── 数值常数 ──────────────────────────────────────────────────────────────────
EPSILON  = 1e-8
MIN_PROB = 0.1
ALNS_RHO = 0.5

# ── 固定基准概率（SC-LLM-OS偏置起点，消除累积漂移）─────────────────────────
OPERATOR_VERSION = "ops_v2"

BASE_PROBS = {
    "worst":         0.22,
    "segment":       0.17,
    "random":        0.16,
    "shaw":          0.17,
    "long_edge":     0.18,
    "multi_segment": 0.10,
}
DESTROY_OPS = list(BASE_PROBS.keys())

REPAIR_BASE_PROBS = {
    "greedy":       0.45,
    "farthest":     0.25,
    "random_order": 0.20,
    "regret2":      0.10,
}
REPAIR_OPS = list(REPAIR_BASE_PROBS.keys())

# ── 并行配置 ──────────────────────────────────────────────────────────────────
SC_ABLATION_STRATEGIES = ["sc_rule_os", "sc_random_os", "sc_fallback_os"]
NON_LLM_STRATEGIES   = ["baseline", "traditional_alns"] + SC_ABLATION_STRATEGIES
MAX_PARALLEL_WORKERS = 4

# ── 输出目录 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ── 运行标识 ──────────────────────────────────────────────────────────────────
VERSION = "v10.0"
RUN_ID  = f"{VERSION}_{time.strftime('%Y%m%d_%H%M%S')}_{LLM_PROVIDER}"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sc_llm_os_results", RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── matplotlib 中文字体自适应 [EXP-11] ────────────────────────────────────────

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


# ── 缓存存取函数 ──────────────────────────────────────────────────────────────

def _cache_path(instance_name: str) -> str:
    return os.path.join(
        SCRIPT_DIR,
        f".cache_tsp_{instance_name}_opt2_{USE_TWO_OPT}_{OPERATOR_VERSION}_{LLM_TRIGGER_ONLY_ON_STAGNATION}.json",
    )


def load_cache(instance_name: str) -> dict:
    cache_path = _cache_path(instance_name)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(instance_name: str, cache_data: dict):
    cache_path = _cache_path(instance_name)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
