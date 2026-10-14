"""向量化封装:智谱 embedding-3(OpenAI 兼容端点,和聊天模型同一个 base_url)。

教学点1【embedding 是什么】:把文本映射成高维空间里的一个坐标,
语义相近的文本坐标相近——"找相似新闻"就变成"找坐标最近的邻居"。
教学点2【维度与成本的权衡】:embedding-3 支持 256~2048 维,
维度越高表达力越强、存储计算越贵。语料小,512 维够用——"够用就好"。
"""
from openai import OpenAI

from config import GLM_API_KEY, GLM_BASE_URL

_DIMENSIONS = 512
_client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def embed(texts: list[str], batch: int = 10) -> list[list[float]]:
    """批量向量化。单次请求有长度上限,分批;一次传多条省请求次数。
    返回顺序与输入顺序一致(API 保证)。"""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = _client.embeddings.create(
            model="embedding-3", input=texts[i : i + batch], dimensions=_DIMENSIONS
        )
        out.extend(d.embedding for d in resp.data)
    return out


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
