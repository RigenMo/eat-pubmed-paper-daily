from datetime import datetime

from tests.canned_responses import make_sample_paper
from tests.test_executor import _delivery_executor
from zotero_arxiv_daily.protocol import CorpusPaper


def prepare(config, monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)
    config.recommendations.enabled = True
    config.recommendations.require_cards = True
    config.recommendations.max_papers = 10
    config.recommendations.focus_mode = mode
    config.executor.max_paper_num = 100
    config.executor.debug = False
    papers = []
    for index in range(16):
        focus = index < 8
        papers.append(make_sample_paper(
            title=("UK Biobank multimorbidity " if focus else "Other topic ") + str(index),
            abstract="Research material with specific disease and design evidence.",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{index+1}/",
            journal=f"Journal {index % 4 if focus else 4 + index % 4}",
            publication_date=f"2026-10-{30-index:02d}",
            journal_metrics={"jif": 12},
        ))
    executor = _delivery_executor(config, papers)
    executor._search_mode = 2
    executor.fetch_zotero_corpus = lambda: [CorpusPaper("Anchor", "Abstract", datetime(2026, 1, 1), [])]
    executor.filter_corpus = lambda corpus: corpus
    monkeypatch.setattr("zotero_arxiv_daily.executor.enrich_journal_metrics", lambda candidates, *args: candidates)
    verified = []
    def verify(candidates, *args, **kwargs):
        verified.extend(candidates)
        for paper in candidates:
            paper.recommendation_matches = [{"reason": "Verified library relationship", "anchor_title": "Anchor"}]
        return candidates
    monkeypatch.setattr("zotero_arxiv_daily.executor.verify_recommendations", verify)
    def assess(candidates, *args, **kwargs):
        for paper in candidates:
            paper.innovation_assessment = {"type": "cognitive", "reason": "Reframes the problem", "evidence": "Research material", "scope": "abstract"}
        return candidates
    monkeypatch.setattr("zotero_arxiv_daily.executor.assess_innovation", assess)
    def reader(paper, *args):
        paper.abstract_zh = "完整摘要译文"
        paper.reading_card = {"essence": ["完整精读卡"]}
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_tldr", reader)
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_title_translation", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: (_ for _ in ()).throw(AssertionError("no SMTP in preflight")))
    return executor, verified


def test_ten_complete_reports_keep_other_topics_and_separate_focus_basis(config, monkeypatch, tmp_path):
    executor, verified = prepare(config, monkeypatch, tmp_path, "ukb_multimorbidity")
    assert executor.run(send_mail=False) == 10
    focus = [paper for paper in executor.last_papers if paper.title.startswith("UK Biobank")]
    assert len(focus) == 6
    assert all(not paper.title.startswith("UK Biobank") for paper in verified)
    assert all(paper.recommendation_matches[0]["basis"] == "current_focus" for paper in focus)
    decisions = executor.selection_audit["selection"]
    assert sum(item["selected"] for item in decisions) == 10
    assert any("focus_limit" in item["skip_reasons"] for item in decisions)
    exported = (tmp_path / "outputs" / "recommendation-audit-summary.json").read_text(encoding="utf-8")
    assert "pubmed.ncbi.nlm.nih.gov" not in exported
    assert "Anchor" not in exported and "candidate_quote" not in exported


def test_balanced_mode_restores_library_gate_for_every_candidate(config, monkeypatch, tmp_path):
    executor, verified = prepare(config, monkeypatch, tmp_path, "balanced")
    assert executor.run(send_mail=False) == 10
    assert len(verified) == 16
    assert all("basis" not in paper.recommendation_matches[0] for paper in executor.last_papers)


def test_library_filter_logs_counts_without_private_titles_or_paths(monkeypatch):
    from types import SimpleNamespace
    from zotero_arxiv_daily.executor import Executor

    executor = Executor.__new__(Executor)
    executor.include_path_patterns = ['PRIVATE_FOLDER']
    executor.ignore_path_patterns = None
    messages = []
    monkeypatch.setattr('zotero_arxiv_daily.executor.glob_match', lambda *args: True)
    monkeypatch.setattr('zotero_arxiv_daily.executor.logger', SimpleNamespace(info=messages.append))
    corpus = [CorpusPaper('PRIVATE_TITLE', 'PRIVATE_ABSTRACT', datetime(2026, 1, 1), ['PRIVATE_FOLDER'])]
    assert executor.filter_corpus(corpus) == corpus
    assert messages and all('PRIVATE_' not in message for message in messages)
