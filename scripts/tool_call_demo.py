"""D2 核心演示:function calling 三段舞步(单轮版)。

D3 我们会把它升级成完整 agent loop(模型可以连续多次调工具直到答完)。
先跑通单轮,看清每一步发生了什么。

运行: uv run python -m scripts.tool_call_demo
"""
from openai import OpenAI

from config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL
from tools.market import TOOL_DEFS, execute


def main() -> None:
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
    messages = [
        {
            "role": "system",
            "content": "你是一个A股投研助手。回答任何数据问题前必须先调用工具查库,"
            "回答中要引用工具返回的具体数字,不要凭记忆编造行情。",
        },
        {"role": "user", "content": "贵州茅台最近10个交易日的行情数据"},
    ]

    # --- 第一段:模型看到工具菜单 + 问题,输出"调用请求"(只是请求!) ---
    resp = client.chat.completions.create(
        model=GLM_MODEL, messages=messages, tools=TOOL_DEFS
    )
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print("模型没有调用工具,直接回答了:")
        print(msg.content)
        return

    print("=== ① 模型请求调用的工具 ===")
    for tc in msg.tool_calls:
        print(f"    {tc.function.name}({tc.function.arguments})")

    # --- 第二段:执行权在我们手里 ---
    messages.append(msg)  # 协议要求:先记录模型的请求,再附上执行结果
    for tc in msg.tool_calls:
        result = execute(tc.function.name, tc.function.arguments)
        print(f"\n=== ② 执行结果(截取前300字) ===\n{result[:300]}")
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )

    # --- 第三段:工具结果喂回去,模型生成面向用户的回答 ---
    final = client.chat.completions.create(
        model=GLM_MODEL, messages=messages, tools=TOOL_DEFS
    )
    print("\n=== ③ 最终回答 ===")
    print(final.choices[0].message.content)


if __name__ == "__main__":
    main()
