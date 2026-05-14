# =============================================================================
#  模块六：LLM 大脑
# =============================================================================

import json
import time
import urllib.request

try:
    from google import genai
except ImportError:
    genai = None

from .config import (
    LLM_PROVIDER,
    API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MIMO_API_KEY,
    MIMO_BASE_URL,
    MIMO_MODEL,
    DESTROY_OPS,
    REPAIR_OPS,
)


def ask_llm_for_mode(meta_state_json: str) -> dict:
    """
    调用API 获取宏观战术模式决策。
    最多重试3次（指数退避），返回值含 is_fallback 字段标记数据质量。
    """
    prompt = f"""你是一个高级组合优化算法的宏观战略指挥官（Strategic Commander）。
你的任务是阅读当前 TSP/ALNS 搜索状态，做因果归因，并同时选择：
1) 宏观战术模式 mode；
2) 本阶段应重点偏置的破坏算子 destroy_focus；
3) 本阶段应重点偏置的修复算子 repair_focus。

注意：你只做“策略偏置”判断，不直接给概率，不生成路线，不解释算法背景。

【当前系统状态 JSON】
{meta_state_json}

【战术模式语义定义】
- exploit          (开发模式): 局部精细改良。适用：best仍在下降，worst/greedy类算子有效。
- explore_topology (拓扑探索): 重构路径结构。适用：structural_anomaly_ratio偏高、长边/片段问题明显。
- explore_global   (全局探索): 强化扰动跳出局部最优。适用：深度停滞、阶段零改善、多个算子失效。

【破坏算子 destroy_focus 可选项】
- worst         : 移除对当前路径代价贡献最高的节点，适合 exploit。
- segment       : 移除单个连续片段，适合局部结构重排。
- random        : 随机移除节点，适合深度停滞时全局扰动。
- shaw          : 移除空间相关的一组节点，适合局部区域重构。
- long_edge     : 移除异常长边端点，适合拓扑异常较高时。
- multi_segment : 移除多个短片段，适合比segment更强、比random更有结构的扰动。

【修复算子 repair_focus 可选项】
- greedy       : 最小增量插入，稳定、偏 exploit。
- farthest     : 最远节点优先插入，避免难插节点留到最后。
- random_order : 随机顺序贪心插入，提升多样性。
- regret2      : regret-2 插入，质量更稳但更重，适合中小规模或拓扑重构阶段。

【决策规则】
- 如果 phase_distance_drop=0 且 stagnation_steps 很高，避免选择 exploit。
- 如果 structural_anomaly_ratio 高或长边问题明显，优先考虑 long_edge/segment/multi_segment + regret2/farthest。
- 如果所有算子 success_rate 都低，优先 explore_global + random/multi_segment + random_order/farthest。
- 如果某个算子 used 很少，不要过度相信它的 success_rate。
- reasoning 必须只用一句话，点名关键指标和选择原因。

严格返回以下JSON，不输出任何其他内容：
{{
    "reasoning": "一句话：指出关键状态指标并说明选择该模式的因果逻辑",
    "mode": "<exploit|explore_topology|explore_global>",
    "destroy_focus": "<worst|segment|random|shaw|long_edge|multi_segment>",
    "repair_focus": "<greedy|farthest|random_order|regret2>"
}}
"""
    max_retry = 3

    for attempt in range(1, max_retry + 1):
        try:
            time.sleep(3)

            if LLM_PROVIDER == "gemini":
                client = genai.Client(api_key=API_KEY)
                res    = client.models.generate_content(
                    model    = "gemini-3.1-pro-preview",
                    contents = prompt,
                    config   = {
                        "response_mime_type": "application/json",
                        "temperature":        0.2,
                    },
                )
                result = json.loads(res.text)

            elif LLM_PROVIDER == "deepseek":
                url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
                }
                data = json.dumps({
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2
                }).encode("utf-8")

                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as response:
                    res_body = response.read().decode("utf-8")
                    content = json.loads(res_body)["choices"][0]["message"]["content"]
                    result = json.loads(content)

            elif LLM_PROVIDER == "mimo":
                url = f"{MIMO_BASE_URL.rstrip('/')}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {MIMO_API_KEY}"
                }
                data = json.dumps({
                    "model": MIMO_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2
                }).encode("utf-8")

                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as response:
                    res_body = response.read().decode("utf-8")
                    content = json.loads(res_body)["choices"][0]["message"]["content"]
                    result = json.loads(content)

            else:
                raise ValueError(f"未知的 LLM_PROVIDER: {LLM_PROVIDER}")

            valid_modes = ("exploit", "explore_topology", "explore_global")
            if result.get("mode") not in valid_modes:
                raise ValueError(f"非法mode字段: {result.get('mode')}")

            default_focus = {
                "exploit":          ("worst", "greedy"),
                "explore_topology": ("long_edge", "regret2"),
                "explore_global":   ("random", "random_order"),
            }
            mode = result["mode"]
            result["destroy_focus"] = result.get("destroy_focus") or default_focus[mode][0]
            result["repair_focus"]  = result.get("repair_focus")  or default_focus[mode][1]

            if result["destroy_focus"] not in DESTROY_OPS:
                raise ValueError(f"非法destroy_focus字段: {result['destroy_focus']}")
            if result["repair_focus"] not in REPAIR_OPS:
                raise ValueError(f"非法repair_focus字段: {result['repair_focus']}")

            result["is_fallback"] = False
            return result

        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON解析失败 (attempt {attempt}): {e}")
        except Exception as e:
            print(f"   ⚠️  API调用失败 (attempt {attempt}): {e}")
            time.sleep(5 * attempt)

    print(f"   🔴 LLM连续失败{max_retry}次，启用Fallback。")
    return {
        "reasoning":   f"Fallback after {max_retry} failures",
        "mode":        "explore_global",
        "destroy_focus": "random",
        "repair_focus":  "random_order",
        "is_fallback": True,
    }
