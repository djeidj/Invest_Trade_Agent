"""D2 预备作业:验证 GLM 的 API key 和模型名真实可用。

运行:
    uv add openai
    uv run python -m scripts.ping_model

教学点:为什么用 openai SDK 而不是智谱官方 SDK?
因为 OpenAI 的接口协议已经是行业事实标准,智谱/DeepSeek/Qwen 都提供
"OpenAI 兼容端点"。用 openai SDK + 改 base_url,将来换模型供应商
代码零改动——这就是 config.py 里模型参数外置的意义。
"""
from openai import OpenAI

from config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL


def main() -> None:
    if not GLM_API_KEY:
        raise SystemExit("GLM_API_KEY 未配置,请检查 .env")

    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
    resp = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": "只回复两个字:收到"}],
    )
    print("模型回复:", resp.choices[0].message.content)
    print("token 用量:", resp.usage.total_tokens)


if __name__ == "__main__":
    main()
