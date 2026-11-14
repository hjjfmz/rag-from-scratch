from rag.bm25 import BM25


def test_chinese_keyword_query_retrieves_matching_document():
    index = BM25()
    index.index([{"text": "知识图谱检索支持故障定位", "metadata": {"source": "graph.md"}},
                 {"text": "购物订单支付", "metadata": {"source": "shop.md"}}])
    hits = index.search("图谱检索")
    assert [row["metadata"]["source"] for row in hits] == ["graph.md"]
