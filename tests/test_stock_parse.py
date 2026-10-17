"""纯逻辑单测:新浪 abstract 解析 + 金额格式化 + 渲染。不联网、不碰存储。"""
import pandas as pd

from tools import stock_tools as st


def test_parse_abstract_merges_metrics_by_report_period():
    # 模拟新浪 abstract:行=指标,列=YYYYMMDD 报告期
    df = pd.DataFrame(
        {
            "选项": ["常用指标", "常用指标", "常用指标"],
            "指标": ["营业总收入", "归母净利润", "经营现金流量净额"],
            "20260630": [9.0e10, 3.3e10, 5.0e10],
            "20260331": [4.5e10, 1.7e10, None],  # 现金流缺失 → None
        }
    )
    rows = st._parse_abstract(df)
    assert len(rows) == 2
    assert rows[0]["report_date"] == "2026-06-30"  # 倒序:最新期在前
    assert rows[0]["revenue"] == 9.0e10
    assert rows[0]["net_profit"] == 3.3e10
    assert rows[1]["report_date"] == "2026-03-31"
    assert rows[1]["op_cashflow"] is None  # 缺失值补 None,不崩


def test_parse_abstract_empty_and_bad_input():
    assert st._parse_abstract(None) == []
    assert st._parse_abstract(pd.DataFrame()) == []
    # 没有"指标"列 → 空
    assert st._parse_abstract(pd.DataFrame({"foo": [1]})) == []


def test_fmt_yi_units_and_missing():
    assert st._fmt_yi(1.23e8) == "1.23亿"
    assert st._fmt_yi(None) == "-"
    assert st._fmt_yi(5000.0) == "5,000"


def test_format_financial_output_shape():
    rows = [
        {"report_date": "2026-06-30", "revenue": 1e10, "net_profit": 2e9, "op_cashflow": 3e9},
        {"report_date": "2026-03-31", "revenue": None, "net_profit": 1e9, "op_cashflow": None},
    ]
    txt = st._format_financial("600519", rows, "测试")
    assert "600519" in txt
    assert "100.00亿" in txt
    assert "营业总收入=-" in txt  # None → 占位符
    assert st._format_financial("000001", [], "测试").startswith("未获取到")
