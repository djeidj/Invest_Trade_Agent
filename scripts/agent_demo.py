"""D3 验收:复合问题,观察 agent 自主拆解、连续调工具。

运行: uv run python -m scripts.agent_demo
"""
from openai import OpenAI

from agent import run_agent
from config import GLM_API_KEY, GLM_BASE_URL


def main() -> None:
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

    question = "对比贵州茅台和平安银行最近5个交易日的收盘价走势,再各挑一条最近的重要新闻。"
    answer = run_agent(client, question)
    print("\n=== 最终回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
