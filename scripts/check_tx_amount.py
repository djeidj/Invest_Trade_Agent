"""验证腾讯源 turnover/amount 字段的真实语义。

背景:官方文档说腾讯源只有 6 列(amount=成交量),实测有 8 列,文档自相矛盾。
原则:字段名 != 字段含义,映射前必须用数据验证。

方法1【量纲假设检验】:
    成交额(元) ≈ 收盘价 × 成交股数;1 手 = 100 股
    → 若 amount 是成交额且 volume 单位为手: amount/(close×volume) ≈ 100
    → 若 amount 是成交量(股)且 volume 单位为手: amount/volume ≈ 100
    → 若 turnover 是换手率(%): 数值会是个位数小数,一眼识别

方法2【交叉校验】:与东财同日数据并排比对(东财字段语义已验证)。

运行: uv run python -m scripts.check_tx_amount
注意用"不复权"数据——复权价会破坏 价格×数量≈金额 的关系。
"""
import akshare as ak

SYMBOL = "600519"  # 茅台:金额巨大,量纲一眼可辨
START, END = "20260921", "20261007"  # 最近几个交易日(10月国庆休市,取到9月底)


def main() -> None:
    tx = ak.stock_zh_a_hist_tx(symbol=SYMBOL, start_date=START, end_date=END, adjust="")
    print("腾讯源原始列:", list(tx.columns))
    print(tx.to_string(index=False), "\n")

    last = tx.iloc[-1]
    close, vol = float(last["close"]), float(last["volume"])
    print(f"=== 量纲假设检验(最新交易日 {last['date']}) ===")
    print(f"close={close}, volume={vol}")
    for col in ("turnover", "amount"):
        if col in tx.columns:
            r_amount = float(last[col]) / (close * vol)  # ≈100 → 该列是成交额(元)
            r_shares = float(last[col]) / vol  # ≈100 → 该列是成交量(股)
            print(f"  {col}: /(close×volume) = {r_amount:.2f}   /volume = {r_shares:.2f}")

    print("\n=== 交叉校验:与东财同日比对 ===")
    try:
        em = ak.stock_zh_a_hist(symbol=SYMBOL, period="daily", start_date=START, end_date=END, adjust="")
        m = tx.merge(em.rename(columns={"日期": "date"}), on="date", how="inner")
        print(m[["date", "成交量", "成交额", "volume", "turnover", "amount"]].to_string(index=False))
        print("\n判读:tx 的哪一列 ≈ 东财.成交额,哪一列就真是成交额")
    except Exception as e:  # noqa: BLE001 东财今天间歇抽风
        print(f"(东财对照不可用: {type(e).__name__} —— 用上面的比值判读即可)")


if __name__ == "__main__":
    main()
