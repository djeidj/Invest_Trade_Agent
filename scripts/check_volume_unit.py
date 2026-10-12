"""成交量单位侦探:用 amount/(close×volume) 比值检验库里 volume 的单位。

原理:成交额(元) ≈ 收盘价 × 成交股数(用收盘价近似均价,有±3%误差)
判读: ratio≈1 → 股(正确)  |  ratio≈100 → 手  |  ratio≈0.01 → 被错误放大100倍的脏数据

运行: uv run python -m scripts.check_volume_unit
"""
import sqlite3

from config import DB_PATH

QUERY = (
    "SELECT date, close, volume, amount,"
    " ROUND(amount / (close * volume), 2) AS ratio"
    " FROM daily_bar"
    " WHERE symbol = ? AND amount IS NOT NULL AND amount > 0 AND volume > 0"
    " ORDER BY date DESC LIMIT 8"
)


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(QUERY, ("600519",)).fetchall()
    print(f"{'date':<12}{'close':>10}{'volume':>15}{'amount':>18}{'ratio':>8}")
    for d, close, vol, amt, ratio in rows:
        print(f"{d:<12}{close:>10.2f}{vol:>15.0f}{amt:>18.0f}{ratio:>8}")
    print("\n判读: ratio≈1 → 股(正确) | ≈100 → 手 | ≈0.01 → 放大100倍的脏数据")


if __name__ == "__main__":
    main()
