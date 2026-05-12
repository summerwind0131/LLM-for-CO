# =============================================================================
#  模块二：数据加载
# =============================================================================

import os
import re
import gzip
import urllib.request

import numpy as np

from .config import SCRIPT_DIR


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
