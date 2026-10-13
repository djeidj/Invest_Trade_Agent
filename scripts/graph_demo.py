"""D4 验收①:同一问题跑两个版本,确认行为等价(祛魅的实证)。

运行: uv run python -m scripts.graph_demo
"""
from openai import OpenAI

from agent import run_agent
from agent.graph import run_graph_agent
from config import GLM_API_KEY, GLM_BASE_URL

QUESTION = "贵州茅台最近5个交易日走势怎么样?"


def main() -> None:
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

    print("======== ① 手写版 run_agent ========")
    a = run_agent(client, QUESTION)

    print("\n======== ② LangGraph 版 run_graph_agent ========")
    b = run_graph_agent(QUESTION)

    print("\n=== 两版回答节选(内容应等价) ===")
    print("[手写]", a[:120].replace("\n", " "), "...")
    print("[框架]", b[:120].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
