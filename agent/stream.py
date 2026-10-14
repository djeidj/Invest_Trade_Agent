"""agent 事件流引擎 —— D6 的心脏:把"最后才给答案"变成"边跑边吐"。

教学点1【引擎与传输分离】:
run_agent_stream 是一个"事件生成器"(generator),yield 一个个事件字典,
完全不关心事件走 HTTP、WebSocket 还是被打印到控制台。
app/main.py 只做一件事:把事件字典包成 SSE 帧推给浏览器。
好处:传输层可替换,引擎一行不改(今天的检查点3)。

教学点2【流式协议的变化】(今天最重要的新知识):
D2-D5 的 create() 等完整响应一次返回;stream=True 返回"分片迭代器",
模型每生成一小段就推一个 chunk:
- 文字: chunk.choices[0].delta.content 是增量文字 → 来多少转发多少,
  前端立刻呈现"打字机"效果
- 工具: delta.tool_calls 是"碎片"——首帧带 id 和 name,
  后续帧只带 arguments 的 JSON 子串,必须按 index 累积(+=),
  收完才能执行。官方文档明确要求拼接(和 OpenAI 格式对齐)。
  这和 D2 的结论同源:模型本质在"吐字符串",流式只是把字符串切碎了吐。

教学点3【熔断逻辑原样保留】:
流式不改变 ReAct 循环结构:有 tool_calls → 执行回填再循环;
没有 → 最终回答。max_steps 熔断照样要有,只是强制总结也改成了流式。
"""
from openai import OpenAI

from agent.loop import SYSTEM_PROMPT
from config import GLM_MODEL
from tools.market import TOOL_DEFS, execute


def run_agent_stream(client: OpenAI, question: str, max_steps: int = 8):
    """以事件流方式运行 agent,逐个 yield 事件字典。协议:
      {"type": "tool_call",   "name":..., "args":...}   模型请求调工具
      {"type": "tool_result", "name":..., "preview":...} 工具结果(前120字)
      {"type": "token",       "text":...}                回答的增量文字
      {"type": "final",       "text":...}                完整回答(收尾用)
      {"type": "error",       "text":...}                出错说明
    """
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        for _ in range(1, max_steps + 1):  # 步数熔断计数,循环体内用不到所以用 _ 占位
            stream = client.chat.completions.create(
                model=GLM_MODEL, messages=messages, tools=TOOL_DEFS, stream=True
            )
            content_parts: list[str] = []
            tool_calls: dict[int, dict] = {}  # index -> {"id","name","arguments"}

            for chunk in stream:
                if not chunk.choices:  # 个别厂商会推无 choices 的收尾帧,跳过
                    continue
                delta = chunk.choices[0].delta
                # 文字增量:直接转发,前端即时显示
                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "token", "text": delta.content}
                # 工具碎片:按 index 归位累积。注意必须用 += 拼接——
                # arguments 一次只到一小片,直接赋值会丢掉之前的片段
                for tc in delta.tool_calls or []:
                    slot = tool_calls.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

            if not tool_calls:  # 没有工具请求 = 最终回答已经流完了
                yield {"type": "final", "text": "".join(content_parts)}
                return

            # 协议要求:先把模型的调用请求记入历史,再紧跟着放结果
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": s["id"],
                            "type": "function",
                            "function": {"name": s["name"], "arguments": s["arguments"]},
                        }
                        for s in tool_calls.values()
                    ],
                }
            )
            for s in tool_calls.values():
                yield {"type": "tool_call", "name": s["name"], "args": s["arguments"]}
                result = execute(s["name"], s["arguments"])
                yield {"type": "tool_result", "name": s["name"], "preview": result[:120]}
                messages.append(
                    {"role": "tool", "tool_call_id": s["id"], "content": result}
                )

        # --- 熔断:步数用尽,不带 tools 强制总结(也用流式转发) ---
        yield {"type": "tool_call", "name": "(已达步数上限,强制总结)", "args": ""}
        stream = client.chat.completions.create(
            model=GLM_MODEL,
            messages=messages
            + [{"role": "user", "content": "请基于以上已查到的信息直接回答,不要再调用工具。"}],
            stream=True,
        )
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
                yield {"type": "token", "text": chunk.choices[0].delta.content}
        yield {"type": "final", "text": "".join(parts)}
    except Exception as e:  # noqa: BLE001 出错也要以事件形式告诉前端,不能让流悄悄断掉
        yield {"type": "error", "text": f"agent 运行出错: {e}"}
