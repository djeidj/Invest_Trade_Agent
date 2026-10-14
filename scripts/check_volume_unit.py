"""成交量单位侦探:用 amount/(close×volume) 比值检验库里 volume 的单位。

原理:成交额(元) ≈ 收盘价 × 成交股数(用收盘价近似均价,有±3%误差)
判读: ratio≈1 → 股(正确)  |  ratio≈100 → 手  |  ratio≈0.01 → 被错误放大100倍
升级版:自动体检全股票池,一行一个结论。

运行: uv run python -m scripts.check_volume_unit
"""
import sqlite3

from config import DB_PATH, WATCHLIST

QUERY = (
    "SELECT date, close, volume, amount,"
    " ROUND(amount / (close * volume), 2) AS ratio"
    " FROM daily_bar"
    " WHERE symbol = ? AND amount IS NOT NULL AND amount > 0 AND volume > 0"
    " ORDER BY date DESC LIMIT 8"
)


def main() -> None:
    all_ok = True
    with sqlite3.connect(DB_PATH) as conn:
        for symbol, name in WATCHLIST.items():
            rows = conn.execute(QUERY, (symbol,)).fetchall()
            if not rows:
                print(f"{name}({symbol}): 库中无数据")
                all_ok = False
                continue
            bad = [(r[0], r[4]) for r in rows if not (0.9 <= r[4] <= 1.1)]
            if bad:
                all_ok = False
                print(f"{name}({symbol}): [BAD] 异常行(疑似单位错) {bad}")
            else:
                print(f"{name}({symbol}): [OK] {len(rows)}行 ratio全部约等于1, 单位=股")
    print("\n结论:", "全库单位统一为股,数据可信" if all_ok else "存在单位异常,先修数据再继续!")


if __name__ == "__main__":
    main()
