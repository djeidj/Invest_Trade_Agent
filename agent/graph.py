"""LangGraph 版 agent —— 和手写 run_agent 完全等价,专门用来给框架"祛魅"。

逐行对照表(面试可直接讲):
| 手写 loop.py                  | LangGraph                                   |
|------------------------------|---------------------------------------------|
| while 循环:执行工具→回填→再问 | 边的环路: agent → tools → agent             |
| if not msg.tool_calls: break | 条件边 router: 有tool_calls→tools, 否则→END |
| messages 列表,手动 append    | State(MessagesState): add_messages 自动追加 |
| execute() 执行工具           | tools 节点(官方预置版叫 ToolNode)           |
| max_steps 熔断               | recursion_limit(框架级熔断)                 |

结论:node 就是函数,edge 就是控制流,state 就是那个 messages 列表。
框架买到的是标准化、可视化、checkpoint 持久化、人机协作钩子——而不是更强的智能。
"""
# 三方导入分三拨,各管一件事:
# ToolMessage      —— LangChain 的消息对象,等价于我们手写版里 {"role": "tool", ...} 那个字典
# ChatOpenAI       —— LangChain 的模型封装,等价于我们手写的 OpenAI(api_key, base_url)
# StateGraph 等    —— LangGraph 本体:建图用的四个零件
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, START, StateGraph

from agent.loop import SYSTEM_PROMPT
from config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL
from tools.market import TOOL_DEFS, execute_dict

# 模型封装。ChatOpenAI 内部干的事和手写版一样:拿 key/base_url 建 HTTP 客户端,
# 只是把"发消息、解析响应"包成了 .invoke(messages) 一个方法。
llm = ChatOpenAI(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
# bind_tools = 手写版里 create(..., tools=TOOL_DEFS) 中的那个 tools= 参数:
# 把"工具菜单"绑到模型上。绑定后模型才能在响应里返回 tool_calls。
llm_with_tools = llm.bind_tools(TOOL_DEFS)


def agent_node(state: MessagesState):
    """agent 节点 = 手写版"把历史发给模型、拿回响应"那一步。

    参数 state:就是当前的"全局状态",形状由下面的 MessagesState 约定——
    实际上就是一个 {"messages": [...]} 字典,里面是到目前为止的全部消息。
    返回值:一个字典 {"messages": [新消息]}。

    【关键理解】为什么只返回"新增的一条"而不是全部历史?
    LangGraph 会用 MessagesState 内置的 add_messages reducer 把返回值
    "合并"进状态——对 messages 来说合并 = 追加(append)。
    也就是说:手写版我们手动 messages.append(msg) 这一步,
    在框架里由 reducer 自动完成。这就是"状态自动管理"的真相。
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tools_node(state: MessagesState):
    """tools 节点 = 手写版的 for tc in msg.tool_calls: execute(...) 那段循环。

    state["messages"][-1] 是刚从 agent 节点出来的 AIMessage,
    它的 .tool_calls 里可能有一条或多条调用请求(还记得茅台那次的4条并发吗)。
    对每条请求:
      tc["name"] → 工具名(如 "get_daily_bars")
      tc["args"] → 参数字典(注意!框架已经替我们 json.loads 好了,
                   所以这里直接用 execute_dict,不用再过一遍 json 解析)
      tc["id"]   → 调用ID,回填结果时带上,让结果能"对号入座"
    ToolMessage(content=结果, tool_call_id=id) 就是手写版的
    {"role": "tool", "tool_call_id": ..., "content": ...} 的对象化写法。
    """
    last = state["messages"][-1]
    return {
        "messages": [
            ToolMessage(
                content=execute_dict(tc["name"], tc["args"]),
                tool_call_id=tc["id"],
            )
            for tc in last.tool_calls
        ]
    }


def router(state: MessagesState) -> Literal["tools", END]:  # type: ignore[type-arg]
    """条件边 = 手写版的 if not msg.tool_calls: break。

    它不是节点,而是一个"路由函数":看完当前状态,返回【下一个节点的名字】。
    - 返回 "tools" → 走 tools 节点(模型还想调工具)
    - 返回 END     → 整个图结束(模型给出最终回答了)
    Literal 标注只是给类型检查器看的:"返回值只可能是这两个"。
    """
    return "tools" if state["messages"][-1].tool_calls else END


# ---- 建图:把上面的零件装配成一张"流程图" ----
builder = StateGraph(MessagesState)
# StateGraph(MessagesState):声明这张图的状态形状 = MessagesState
# (即 "messages 键 + add_messages 追加式合并规则")
builder.add_node("agent", agent_node)  # 注册节点:名字 → 函数
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")  # 固定边:流程从 START 进入 agent(循环初始化)
builder.add_conditional_edges("agent", router)  # 动态边:agent 之后走哪,router 说了算(= break)
builder.add_edge("tools", "agent")  # 固定边:tools 干完回 agent(= while 循环本体)
graph = builder.compile()  # compile:把装配图冻结成可运行对象(校验拓扑、预编译)


def run_graph_agent(question: str, verbose: bool = True) -> str:
    """跑图并返回最终回答。

    graph.stream(inputs):不是流式输出文字!而是"每执行完一个超步(super-step),
    就把【完整状态】吐给你一次"。一个超步 ≈ 手写版 for 循环的一轮:
    agent 出手一次 + tools 回填一次 = 一个超步。
    这就是框架白送的可观测性:手写版的每步 print,框架用事件流实现。

    config={"recursion_limit": 16}:框架级熔断 = 手写版 max_steps。
    区别:超限时框架直接抛 GraphRecursionError 异常,
    而手写版会优雅降级(强制总结)。生产中要 try/except 接住再兜底。
    """
    inputs = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    }
    final = None
    for state in graph.stream(inputs, stream_mode="values", config={"recursion_limit": 16}):
        last = state["messages"][-1]  # 这个超步里最新产生的那条消息
        if isinstance(last, ToolMessage):  # 工具结果刚回填
            if verbose:
                print(f"          -> {last.content[:120]}...")
        elif getattr(last, "tool_calls", None):  # 模型刚发出调用请求
            if verbose:
                for tc in last.tool_calls:
                    print(f"[graph] 调用 {tc['name']}({tc['args']})")
        final = last  # 最后一次循环时,last 必是不带 tool_calls 的最终回答
    return final.content
