"""D5 验收:语义检索 + 引用作答。

问题故意选了一个关键词不易精确匹配的场景,体会语义检索
和 get_stock_news(关键词型工具)的差别。

运行: uv run python -m scripts.rag_demo
"""
from openai import OpenAI

from agent import run_agent
from config import GLM_API_KEY, GLM_BASE_URL


def main() -> None:
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)

    # question = "语料里有没有关于i茅台APP维护升级的报道?原文怎么说的?"
    question = "语料里有没有关于矿泉水行业的报道？"
    answer = run_agent(client, question)
    print("\n=== 最终回答(检查是否带[编号]引用与来源) ===")
    print(answer)


if __name__ == "__main__":
    main()
