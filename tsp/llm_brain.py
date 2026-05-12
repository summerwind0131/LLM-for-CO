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
)


def ask_llm_for_mode(meta_state_json: str) -> dict:
    """
    调用API 获取宏观战术模式决策。
    最多重试3次（指数退避），返回值含 is_fallback 字段标记数据质量。
    """
    prompt = f"""你是一个高级组合优化算法的宏观战略指挥官（Strategic Commander）。
你的任务是阅读当前搜索状态，基于算子历史表现进行因果归因，选择最合适的宏观战术模式。

【当前系统状态 JSON】
{meta_state_json}

【战术模式语义定义】
- exploit          (开发模式): 侧重精细微调（偏置worst算子）。
                               适用：搜索平稳下降，worst算子success_rate较高时。
- explore_topology (拓扑探索): 侧重切断路径中的交叉长边（偏置segment算子）。
                               适用：structural_anomaly_ratio较高，陷入局部最优时。
- explore_global   (全局探索): 引入彻底随机扰动（偏置random算子）。
                               适用：深度停滞，所有算子均失效时。



严格返回以下JSON，不输出任何其他内容：
{{
    "reasoning": "一句话：指出关键状态指标并说明选择该模式的因果逻辑",
    "mode": "<exploit|explore_topology|explore_global>"
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

            if result.get("mode") not in (
                "exploit", "explore_topology", "explore_global"
            ):
                raise ValueError(f"非法mode字段: {result.get('mode')}")
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
        "is_fallback": True,
    }
