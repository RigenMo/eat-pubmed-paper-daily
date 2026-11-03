from datetime import datetime

from tests.canned_responses import make_sample_paper
from tests.test_executor import _delivery_executor
from zotero_arxiv_daily.protocol import CorpusPaper


def prepare(config, monkeypatch, journals, failed=()):
    config.recommendations.enabled = True
    config.recommendations.require_cards = True
    config.recommendations.max_papers = 5
    config.recommendations.max_per_journal = 2
    config.executor.max_paper_num = 100
    config.executor.debug = False
    papers = [
        make_sample_paper(title=f"Paper {i}", journal=journal, publication_date=f"2026-10-{31-i:02d}")
        for i, journal in enumerate(journals)
    ]
    executor = _delivery_executor(config, list(reversed(papers)))
    executor._search_mode = 2
    executor.fetch_zotero_corpus = lambda: [CorpusPaper("Anchor", "Abstract", datetime(2026, 1, 1), [])]
    executor.filter_corpus = lambda corpus: corpus
    monkeypatch.setattr("zotero_arxiv_daily.executor.enrich_journal_metrics", lambda candidates, *args: candidates)
    monkeypatch.setattr("zotero_arxiv_daily.executor.verify_recommendations", lambda candidates, *args, **kwargs: candidates)

    def reader(paper, *args):
        if paper.title not in failed:
            paper.reading_card = {"essence": ["完整卡片"]}

    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_tldr", reader)
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_title_translation", lambda *args: None)
    rendered = []
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda papers, summary=None: rendered.append((list(papers), summary)) or "HTML")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: None)
    return executor, papers, rendered


def test_latest_five_complete_cards_are_selected(config, monkeypatch):
    executor, papers, _ = prepare(config, monkeypatch, ["A", "A", "B", "B", "C", "C", "D"])
    assert executor.run(send_mail=False) == 5
    assert executor.last_papers == papers[:5]


def test_failed_card_is_replaced_without_consuming_journal_quota(config, monkeypatch):
    executor, papers, _ = prepare(config, monkeypatch, ["A", "A", "A", "B", "B", "C", "C"], {"Paper 0"})
    assert executor.run(send_mail=False) == 5
    assert executor.last_papers == papers[1:6]


def test_shortage_is_disclosed_at_start_of_summary(config, monkeypatch):
    executor, papers, rendered = prepare(config, monkeypatch, ["A", "B", "C"])
    executor.generate_email_summary = lambda candidates: "今日摘要"
    assert executor.run(send_mail=False) == 3
    assert executor.last_papers == papers
    assert rendered[0][1].startswith("3/5") or rendered[0][1].split("今日摘要")[0].find("3/5") >= 0
