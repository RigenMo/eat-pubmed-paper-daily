from types import SimpleNamespace

from zotero_arxiv_daily import journal_metrics
from zotero_arxiv_daily.construct_email import get_block_html
from zotero_arxiv_daily.newsletter_card import complete_newsletter, valid_newsletter_abstract
from zotero_arxiv_daily.publisher_jif import parse_nature_metrics

ABSTRACT = "背景：已知和未知\n目的：目标\n方法：研究设计\n结果：估计\n结论：这项证据的结论"
CARD = {"angle": "兼有", "basis": "具体比较和测量", "intuition": "据摘要提炼：问题",
        "comparison": "A 与 B", "without": "不能分辨", "advance": "结果支持的意义"}


def test_edition_and_two_year_column_are_bound_to_publisher_heading():
    html = """
    <table><tr><td>Journal Impact Factor</td></tr><tr><td>not evidence</td></tr></table>
    <h2 id="citation-impact">2025&nbsp;Journal Metrics</h2>
    <table><tr><td>Journal</td><td>5-year Journal Impact Factor</td><th>Journal Impact Factor</th></tr>
    <tr><td><a>Nature Study</a></td><td>87.5</td><td>12.3</td></tr>
    <tr><td>Nature Empty</td><td>14.1</td><td>-</td></tr></table>
    """
    parsed = parse_nature_metrics(html)
    assert set(parsed) == {"nature study"}
    assert parsed["nature study"]["jif"] == 12.3
    assert parsed["nature study"]["jif_year"] == 2025


def test_dated_gate_rejects_unmatched_unversioned_source(monkeypatch):
    cfg = {"enabled": True, "min_jif": 10, "require_dated_jif": True,
           "easyscholar": {"enabled": True, "api_key": "fixture"},
           "whitelist": [{"name": "Audit Journal", "aliases": ["Audit J"], "year": 2025,
                          "jif": 10.0, "source": "https://publisher.example/metrics"}]}
    queries = []
    monkeypatch.setattr(journal_metrics.EasyScholarClient, "lookup", lambda self, name:
                        queries.append(name) or {"jif": 999.0, "jif_year": None, "jcr_quartile": "Q1"})
    monkeypatch.setattr(journal_metrics, "load_nature_metrics", lambda: {
        "nature verified": {"jif": 12.0, "jif_year": 2025, "source": "https://www.nature.com/example"},
        "nature low": {"jif": 9.9, "jif_year": 2025, "source": "https://www.nature.com/example"},
    })
    a = SimpleNamespace(journal="Audit J")
    b = SimpleNamespace(journal="Nature Verified")
    c = SimpleNamespace(journal="Unknown", journal_metrics={"jif": 99, "source": "https://www.easyscholar.cc/"})
    d = SimpleNamespace(journal="Nature Low")
    assert journal_metrics.enrich_journal_metrics([a, b, c, d], cfg) == [a, b]
    assert a.journal_metrics["jif"] == 10 and b.journal_metrics["jif_year"] == 2025
    assert "Audit Journal" in queries and "Unknown" not in queries
    assert a.journal_metrics.get("jcr_year") is None


def test_card_shows_one_abstract_and_four_independent_metric_badges():
    p = SimpleNamespace(reading_card={"newsletter": CARD}, abstract_zh=ABSTRACT, title_zh="中文题名")
    assert complete_newsletter(p)
    assert not valid_newsletter_abstract(ABSTRACT + "\n结果：重复")
    html = get_block_html("English title", "Author", "", "", "https://example.org/doi",
                          journal="Example", title_zh=p.title_zh, reading_card=p.reading_card,
                          abstract_zh=ABSTRACT, publication_date="2026-11-27",
                          journal_metrics={"jif": 11, "jif_year": 2025, "source": "https://example.org/metrics",
                                           "jcr_quartile": "Q1"})
    assert html.count("中文摘要") == 1
    assert html.count("#fff3cd") == 4 and html.count("未核实") == 3
    assert "11（2025）" in html and "兼有" in html and "研究启发" not in html
    assert "原文链接" in html and "中文题名" in html and "English title" not in html
