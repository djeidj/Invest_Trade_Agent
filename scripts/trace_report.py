"""D7 验收:打印 trace 报告——钱花在哪一步,时间花在哪一步。

先启动 web 聊几轮,再运行: uv run python -m scripts.trace_report
(可选参数:显示最近 N 个会话,如  uv run python -m scripts.trace_report 3)
"""
import sys

from storage import trace as trace_db


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sessions = trace_db.recent_sessions(limit)
    if not sessions:
        print("还没有 trace 记录。先启动 web 并聊几轮。")
        return

    total_cost, total_pt, total_ct = 0.0, 0, 0
    for s in sessions:
        print("=" * 76)
        print(f"[#{s['id']}] {s['ts']}  status={s['status']}")
        print(f"  问: {s['question'][:60]}")
        print(
            f"  tokens: {s['prompt_tokens']}入/{s['completion_tokens']}出"
            f"  成本: {s['cost_yuan']:.6f} 元  会话耗时: {s['duration_ms']}ms"
        )
        spans = trace_db.session_spans(s["id"])
        span_cost = sum(sp["cost_yuan"] for sp in spans) or 1e-12
        span_time = sum(sp["latency_ms"] for sp in spans) or 1
        for sp in spans:
            pct_c = sp["cost_yuan"] / span_cost * 100
            pct_t = sp["latency_ms"] / span_time * 100
            print(
                f"    #{sp['seq']:<2} [{sp['kind']:<4}] {sp['name']:<24}"
                f" {sp['cost_yuan']:.6f}元({pct_c:4.1f}%)"
                f"  {sp['latency_ms']:>6}ms({pct_t:4.1f}%)"
            )
        total_cost += s["cost_yuan"]
        total_pt += s["prompt_tokens"] or 0
        total_ct += s["completion_tokens"] or 0

    print("=" * 76)
    n = len(sessions)
    print(
        f"共 {n} 个会话 | 平均 tokens {total_pt // n}入/{total_ct // n}出"
        f" | 平均成本 {total_cost / n:.6f} 元/次"
    )


if __name__ == "__main__":
    main()
