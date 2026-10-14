"""eval 用例库:能力矩阵 × 检查器。

教学点1【用例怎么选】——覆盖"能力维度"而不是堆数量:
数据问答(单股/多股)、多工具编排、新闻、RAG引用、RAG拒答、错误容忍。
每个维度至少 1 条,这样改提示词/换模型后一跑,就知道**哪类**能力被改坏了,
而不只是"挂了 3 个用例"。

教学点2【标准答案从哪来】——ground truth 一律不写死值:
数字类用例的期望值用 db_close_prices 检查器从数据库现拉,
模型必须真的读库才能答对。手写死值的用例会在数据更新后天天误报,
而且会诱导你把"模型复述了死值"误当成"agent 还在工作"。
"""
CASES: list[dict] = [
    {
        # 能力:单股数据问答 + 数据接地(答的数字必须真是库里的)
        "id": "bars_grounding_moutai",
        "question": "贵州茅台最近5个交易日的收盘价分别是多少?",
        "checks": [
            {"type": "db_close_prices", "symbol": "600519", "days": 5},
            {"type": "contains", "value": "茅台"},
        ],
    },
    {
        # 能力:多股数据问答(要求分别查两只股票)
        "id": "bars_grounding_two_stocks",
        "question": "对比平安银行和紫金矿业最近3个交易日的收盘价",
        "checks": [
            {"type": "db_close_prices", "symbol": "000001", "days": 3},
            {"type": "db_close_prices", "symbol": "601899", "days": 3},
        ],
    },
    {
        # 能力:新闻工具 + 语义级验收(有没有真列出新闻,judge 说了算)
        "id": "news_listing",
        "question": "宁德时代最近有什么新闻?",
        "checks": [
            {
                "type": "llm_judge",
                "criteria": "回答列出了至少一条具体的新闻标题或内容摘要,而不是空泛套话或编造",
            },
        ],
    },
    {
        # 能力:RAG 引用纪律(检索到内容时必须带[编号]+来源)
        "id": "rag_citation_imoutai",
        "question": "语料里有没有关于i茅台APP维护升级的报道?原文怎么说的?",
        "checks": [
            {"type": "citations"},
            {"type": "contains", "value": "i茅台"},
            {
                "type": "llm_judge",
                "criteria": "回答标注了[编号]引用和来源(标题,媒体),并忠实转述了报道内容",
            },
        ],
    },
    {
        # 能力:RAG 防幻觉——语料没有的内容必须明确拒答,不许编
        "id": "rag_refusal_water",
        "question": "语料里有没有关于矿泉水行业的报道?",
        "checks": [
            {"type": "refusal"},
            {
                "type": "llm_judge",
                "criteria": "回答明确说明语料中没有相关内容,没有编造矿泉水行业的报道",
            },
        ],
    },
    {
        # 能力:RAG 防幻觉(第二维度:库外标的)
        "id": "rag_refusal_tesla",
        "question": "语料里有没有特斯拉的新闻?",
        "checks": [
            {"type": "refusal"},
        ],
    },
    {
        # 能力:错误容忍——查不存在的代码,要不崩溃、给引导
        "id": "invalid_symbol_graceful",
        "question": "帮我查一下999999的行情",
        "checks": [
            {"type": "contains_any", "values": ["没有", "查不到", "不存在", "未检索"]},
        ],
    },
    {
        # 能力:多工具编排(一条问题里既查行情又查消息面)
        "id": "multi_tool_bars_and_news",
        "question": "紫金矿业最近5天走势如何?有什么消息面?",
        "checks": [
            {"type": "db_close_prices", "symbol": "601899", "days": 5},
            {
                "type": "llm_judge",
                "criteria": "回答既包含行情数据(价格/涨跌),也提到了新闻或消息面内容",
            },
        ],
    },
]
