"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


def test_executor_passes_configured_llm_timeout_to_openai(config, monkeypatch):
    """The OpenAI-compatible client receives the timeout in seconds."""
    from omegaconf import open_dict

    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    with open_dict(config):
        config.llm.api.timeout = 1234
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", fake_openai)

    Executor(config)

    assert captured["timeout"] == 1234.0


def test_executor_applies_debug_limit_after_reranking(config, monkeypatch):
    """Debug mode limits the ranked delivery set, not PubMed retrieval."""
    from omegaconf import open_dict
    from tests.canned_responses import make_sample_paper, make_stub_openai_client

    with open_dict(config):
        config.executor.debug = True
        config.executor.max_paper_num = 100
        config.executor.source = ["stub"]
        config.search.mode = 5

    class StubRetriever:
        def retrieve_papers(self):
            return [make_sample_paper(title=f"Paper {i}") for i in range(10)]

    class StubReranker:
        def rerank(self, papers, corpus, keyword_query=None, favorite_journals=None):
            assert len(papers) == 10
            return papers

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda corpus: corpus
    executor._search_mode = 5
    executor._keywords_raw = None
    executor._structured_keywords = None
    executor.retrievers = {"stub": StubRetriever()}
    executor.reranker = StubReranker()
    executor.openai_client = make_stub_openai_client()

    generated = []
    monkeypatch.setattr(
        "zotero_arxiv_daily.protocol.Paper.generate_tldr",
        lambda self, client, llm: generated.append(self.title),
    )
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_affiliations", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_title_translation", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda papers, summary=None: len(papers))
    sent = []
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda cfg, body: sent.append(body))

    result = executor.run()

    assert result == 5
    assert len(generated) == 5
    assert sent == [5]


def test_executor_strict_journal_gate_does_not_allow_arxiv_bypass(config, monkeypatch):
    """With the whitelist enabled, non-PubMed candidates cannot reach ranking."""
    from omegaconf import open_dict
    from tests.canned_responses import make_sample_paper

    with open_dict(config):
        config.journals.enabled = True
        config.executor.send_empty = True
        config.search.mode = 5

    class StubRetriever:
        def retrieve_papers(self):
            return [make_sample_paper(title="arXiv bypass candidate")]

    class FailingReranker:
        def rerank(self, *args, **kwargs):
            pytest.fail("non-PubMed candidate must be removed before ranking")

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda corpus: corpus
    executor._search_mode = 5
    executor._keywords_raw = None
    executor._structured_keywords = None
    executor.retrievers = {"arxiv": StubRetriever()}
    executor.reranker = FailingReranker()
    executor.openai_client = object()

    sent = []
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda papers, summary=None: papers)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda cfg, body: sent.append(body))

    result = executor.run()

    assert result == 0
    assert sent == [[]]


# ---------------------------------------------------------------------------
# fetch_zotero_corpus
# ---------------------------------------------------------------------------


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch):
    """Full pipeline: Zotero fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_corpus,
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub pyzotero
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 5. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


# ---------------------------------------------------------------------------
# generate_email_summary — character pool now lives in config/characters.yaml
# ---------------------------------------------------------------------------


def test_generate_email_summary_returns_empty_when_character_pool_empty(config, monkeypatch):
    """Setting executor.character_pool to [] disables the opening summary."""
    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper, make_stub_openai_client

    with open_dict(config):
        config.executor.character_pool = []

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.openai_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.random.choice", lambda pool: pytest.fail("should not be called"))

    summary = executor.generate_email_summary([make_sample_paper()])

    assert summary == ""


def test_generate_email_summary_uses_configured_character_pool(config):
    """The opening summary is voiced by a character drawn from config.executor.character_pool
    and signed off with that character's name."""
    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper, make_stub_openai_client

    with open_dict(config):
        config.executor.character_pool = [{"name": "测试角色", "prompt": "You are a test character."}]

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.openai_client = make_stub_openai_client()

    summary = executor.generate_email_summary([make_sample_paper()])

    assert summary.endswith("—— 测试角色 如是说")


def test_executor_rejects_empty_source_list(config):
    """mix_mode=5 (arxiv-family only) with an empty executor.source list leaves
    no retriever configured; this must fail fast with a clear error instead of
    crashing later inside ThreadPoolExecutor(max_workers=0)."""
    from omegaconf import open_dict

    with open_dict(config):
        config.executor.source = []
        config.source.mix_mode = 5

    with pytest.raises(ValueError, match="No paper retriever was configured"):
        Executor(config)


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


def _delivery_executor(config, papers):
    from types import SimpleNamespace

    executor = Executor.__new__(Executor)
    executor.config = config
    executor._search_mode = 1
    executor._keywords_raw = None
    executor._structured_keywords = None
    executor.retrievers = {"stub": SimpleNamespace(retrieve_papers=lambda: papers)}
    executor.reranker = SimpleNamespace(rerank=lambda candidates, *args, **kwargs: candidates)
    executor.openai_client = object()
    executor.generate_email_summary = lambda candidates: ""
    return executor


def test_all_failed_reading_cards_stop_before_smtp_even_if_translation_exists(config, monkeypatch):
    from tests.canned_responses import make_sample_paper

    config.recommendations.require_cards = True
    paper = make_sample_paper()
    executor = _delivery_executor(config, [paper])

    def partial_reader(self, *args):
        self.abstract_zh = "已经生成的摘要译文。"
        self.reading_card = None
        self.reader_status = "partial"

    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_tldr", partial_reader)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: pytest.fail("must not send failed reports"))

    with pytest.raises(RuntimeError, match="No complete reading cards"):
        executor.run()
    assert paper.abstract_zh == "已经生成的摘要译文。"


def test_no_send_preflight_prepares_the_real_mail_without_smtp(config, monkeypatch):
    from tests.canned_responses import make_sample_paper

    config.recommendations.require_cards = True
    paper = make_sample_paper()
    executor = _delivery_executor(config, [paper])

    def completed_reader(self, *args):
        self.abstract_zh = "译文。"
        self.reading_card = {"essence": ["示例。"]}

    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_tldr", completed_reader)
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_title_translation", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda *args, **kwargs: "prepared HTML")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: pytest.fail("preflight cannot send"))

    assert executor.run(send_mail=False) == 1
    assert executor.last_papers == [paper]
    assert executor.last_email_content == "prepared HTML"


def test_zotero_only_mode_builds_source_queries_before_pubmed_retrieval(config, monkeypatch):
    from types import SimpleNamespace
    from zotero_arxiv_daily.recommendations import build_zotero_queries

    executor = _delivery_executor(config, [])
    executor._search_mode = 5
    reference = CorpusPaper("Dietary patterns and heart failure", "Source abstract.", datetime(2026, 1, 1), [])
    executor.fetch_zotero_corpus = lambda: [reference]
    executor.filter_corpus = lambda corpus: corpus
    injected = []
    def source_queries(corpus, *, openai_client, llm_params):
        assert corpus == [reference]
        assert openai_client is executor.openai_client
        assert llm_params is config.llm
        return build_zotero_queries(corpus)
    monkeypatch.setattr("zotero_arxiv_daily.executor.build_zotero_queries", source_queries)
    executor.retrievers = {"pubmed": SimpleNamespace(
        set_pubmed_queries=lambda queries: injected.extend(queries),
        retrieve_papers=lambda: [],
    )}

    assert executor.run() == 0
    assert injected and all("[tiab]" in query for query in injected)
