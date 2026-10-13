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
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, START, StateGraph

from agent.loop import SYSTEM_PROMPT
from config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL
from tools.market import TOOL_DEFS, execute_dict

llm = ChatOpenAI(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
llm_with_tools = llm.bind_tools(TOOL_DEFS)  # 把工具菜单绑定给模型


def agent_node(state: MessagesState):
    """agent 节点 = 手写版"把历史发给模型、拿回响应"那一步。"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tools_node(state: MessagesState):
    """tools 节点 = 手写版的 for tc in msg.tool_calls: execute(...) 那段循环。"""
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
    """条件边 = 手写版的 if not msg.tool_calls: break。"""
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router)
builder.add_edge("tools", "agent")  # 这条回边就是"while 循环"本体
graph = builder.compile()


def run_graph_agent(question: str, verbose: bool = True) -> str:
    """跑图并返回最终回答。

    用 stream 逐超步(super-step)拿状态——框架内置的"行动轨迹",
    等价于手写版里我们手动 print 的那些日志。
    """
    inputs = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    }
    final = None
    for state in graph.stream(inputs, stream_mode="values", config={"recursion_limit": 16}):
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            if verbose:
                print(f"          -> {last.content[:120]}...")
        elif getattr(last, "tool_calls", None):
            if verbose:
                for tc in last.tool_calls:
                    print(f"[graph] 调用 {tc['name']}({tc['args']})")
        final = last
    return final.content
