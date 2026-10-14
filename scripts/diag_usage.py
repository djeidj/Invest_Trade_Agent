"""诊断:GLM 流式响应里 usage 到底在哪一帧?(实测为准,文档不可信)

跑法: uv run python -m scripts.diag_usage
把输出整段贴回来。
"""
from openai import OpenAI

from config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL

client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def run(with_stream_options: bool) -> None:
    label = "传 stream_options" if with_stream_options else "不传 stream_options"
    print(f"=== {label} ===")
    kwargs = {"stream_options": {"include_usage": True}} if with_stream_options else {}
    stream = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": "用一句话介绍贵州茅台"}],
        stream=True,
        **kwargs,
    )
    n = 0
    seen_usage = 0
    for chunk in stream:
        n += 1
        usage = getattr(chunk, "usage", None)
        if usage:
            seen_usage += 1
        n_choices = len(chunk.choices)
        finish = chunk.choices[0].finish_reason if n_choices else "-"
        mark = "  <== usage!" if usage else ""
        # 只打印关键帧:带 usage 的帧、头两帧、有 finish_reason 的帧,避免刷屏
        if usage or n <= 2 or finish not in (None, "-"):
            print(f"  chunk#{n:02d} choices={n_choices} finish={finish}{mark}")
    print(f"  共 {n} 帧,带 usage 的帧数: {seen_usage}\n")


run(False)
run(True)
