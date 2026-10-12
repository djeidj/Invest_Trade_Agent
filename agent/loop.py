"""agent loop:整个项目的心脏。

教学点1【agent = 受控的 while 循环】:
每轮把完整历史(含所有工具结果)发给模型:
  模型请求工具 → 执行 → 结果回填 → 下一轮
  模型直接回答 → 退出
就这么多。LangGraph/LangChain 的全部价值,都是围绕这个循环做工程化封装。

教学点2【模型为什么能"多步思考"】:
每一轮它都能看到之前所有工具调用和结果,
所以能根据刚查到的信息决定下一步——即 ReAct 模式(Reason→Act→Observe)。

教学点3【必须熔断】:
模型可能无限重试一个报错的工具。max_steps 是失控防护,
熔断后强制模型"基于已有信息作答"(不带 tools 再调一次,它就没法再请求工具了)。
"""
from openai import OpenAI

from config import GLM_MODEL
from tools.market import TOOL_DEFS, execute

SYSTEM_PROMPT = (
    "你是一个A股投研助手。回答任何数据问题前必须先调用工具查库,"
    "回答中要引用工具返回的具体数字,不要凭记忆编造行情。"
    "需要多只股票的数据时逐个查询,全部查完再总结。"
)


def run_agent(
    client: OpenAI,
    question: str,
    max_steps: int = 8,
    verbose: bool = True,
) -> str:
    """运行 agent 直到模型给出最终回答,返回最终回答文本。

    verbose=True 时打印每步行动轨迹——这是最朴素的 tracing,D7 会升级成正式方案。
    """
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=GLM_MODEL, messages=messages, tools=TOOL_DEFS
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:  # 模型不再请求工具 → 它认为信息足够了
            if verbose:
                print(f"[step {step}] 模型给出最终回答")
            return msg.content

        messages.append(msg)  # 协议要求:先记录模型的调用请求,紧接着才是结果
        for tc in msg.tool_calls:  # 一轮里模型可能并发请求多个工具
            result = execute(tc.function.name, tc.function.arguments)
            if verbose:
                print(f"[step {step}] 调用 {tc.function.name}({tc.function.arguments})")
                print(f"          -> {result[:120]}...")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # --- 熔断:步数用尽,不带 tools 强制总结(它想调工具也没得调) ---
    if verbose:
        print(f"[熔断] 已达 {max_steps} 步上限,强制基于已有信息作答")
    resp = client.chat.completions.create(
        model=GLM_MODEL,
        messages=messages
        + [{"role": "user", "content": "请基于以上已查到的信息直接回答,不要再调用工具。"}],
    )
    return resp.choices[0].message.content
