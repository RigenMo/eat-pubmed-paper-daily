"""Tests for corpus-grounded queries and the conservative relevance gate."""

from datetime import datetime
import json
import re
from types import SimpleNamespace

import pytest

from zotero_arxiv_daily.protocol import CorpusPaper
from zotero_arxiv_daily.recommendations import (
    RecommendationQueryError,
    RecommendationVerificationError,
    build_zotero_queries,
    verify_recommendations,
)
from tests.canned_responses import make_sample_paper


def _chat_response(payload) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
    )


class _DeterministicClient:
    def __init__(self, decision_factory):
        self.decision_factory = decision_factory
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user_material = kwargs["messages"][1]["content"]
        ids = re.findall(r"Candidate ID: (\d+)", user_material)
        return _chat_response(self.decision_factory(ids, user_material))


def _anchor(title: str, abstract: str, *, key: str = "A1") -> dict:
    return {
        "anchor_title": title,
        "anchor_url": f"https://www.zotero.org/items/{key}",
        "anchor_key": key,
        "anchor_abstract": abstract,
        "similarity": 0.91,
    }


def _paper_with_anchor() -> object:
    paper = make_sample_paper(
        title="Dietary protein timing and insulin sensitivity",
        abstract=(
            "We test whether dietary protein timing changes insulin sensitivity "
            "after a meal-timing intervention."
        ),
    )
    paper.recommendation_matches = [
        _anchor(
            "Meal timing and metabolic health",
            "Meal timing and dietary protein distribution may shape metabolic health.",
        )
    ]
    return paper


def test_build_zotero_queries_is_deterministic_source_grounded_and_balanced():
    corpus = [
        CorpusPaper(
            title="Cardiac magnetic resonance fibrosis remodeling",
            abstract="",
            added_date=datetime(2026, 1, 1),
            paths=["research/cardiac imaging"],
        ),
        CorpusPaper(
            title="Dietary protein meal timing metabolic health",
            abstract="",
            added_date=datetime(2026, 1, 2),
            paths=["research/nutrition timing"],
        ),
        CorpusPaper(
            title="Gut microbiome metabolite cross talk",
            abstract="",
            added_date=datetime(2026, 1, 3),
            paths=["research/multiomics"],
        ),
    ]

    queries = build_zotero_queries(corpus)
    assert queries == build_zotero_queries(corpus)
    assert queries
    assert all(1 <= len(re.findall(r'"([^"]+)"\[tiab\]', query)) <= 5 for query in queries)

    title_texts = {
        " ".join(re.sub(r"[^\w]+", " ", paper.title.casefold()).split())
        for paper in corpus
    }
    phrases = [phrase.casefold() for query in queries for phrase in re.findall(r'"([^"]+)"\[tiab\]', query)]
    assert phrases
    assert all(any(phrase in title for title in title_texts) for phrase in phrases)
    assert any("cardiac" in query for query in queries)
    assert any("dietary" in query for query in queries)
    assert any("microbiome" in query for query in queries)


def test_build_zotero_queries_covers_more_than_six_titles_by_default():
    corpus = [
        CorpusPaper(
            title=f"Specific topic {index} biomarker pathway",
            abstract="",
            added_date=datetime(2026, 1, 1),
            paths=["collection/shared"],
        )
        for index in range(7)
    ]

    queries = build_zotero_queries(corpus)
    assert len(queries) > 1
    for index in range(7):
        assert any(f"topic {index}" in query for query in queries)


def test_build_zotero_queries_blank_corpus_fails_closed():
    with pytest.raises(ValueError, match="title|topic phrases"):
        build_zotero_queries([])


def test_verify_recommendations_rejects_unrelated_candidates_and_clears_matches():
    paper = _paper_with_anchor()

    def all_false(ids, _):
        return {
            "recommendations": [
                {
                    "id": item_id,
                    "related": False,
                    "anchor_index": None,
                    "reason": "No direct topic relationship in the supplied material.",
                    "candidate_quote": None,
                    "anchor_quote": None,
                }
                for item_id in ids
            ]
        }

    client = _DeterministicClient(all_false)
    assert verify_recommendations([paper], client, {}) == []
    assert paper.recommendation_matches == []
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "decision_factory",
    [
        lambda ids, _: {
            "recommendations": [
                {
                    "id": ids[0],
                    "related": True,
                    "anchor_index": 1,
                    "reason": "Direct relation claimed.",
                    "candidate_quote": "dietary protein",
                    "anchor_quote": "meal timing",
                }
            ]
        },
        lambda ids, _: {
            "recommendations": [
                {
                    "id": ids[0],
                    "related": True,
                    "anchor_index": 0,
                    "reason": "Direct relation claimed.",
                    "candidate_quote": "invented phrase",
                    "anchor_quote": "meal timing",
                }
            ]
        },
    ],
)
def test_verify_recommendations_wrong_anchor_or_quote_fails_closed(decision_factory):
    with pytest.raises(RecommendationVerificationError, match="failed closed"):
        verify_recommendations([_paper_with_anchor()], _DeterministicClient(decision_factory), {})


def test_verify_recommendations_keeps_only_verified_anchor_and_safe_quotes():
    paper = _paper_with_anchor()

    def related(ids, _):
        return {
            "recommendations": [
                {
                    "id": ids[0],
                    "related": True,
                    "anchor_index": 0,
                    "reason": "Both materials directly address dietary timing.",
                    "candidate_quote": "dietary protein timing",
                    "anchor_quote": "meal timing and dietary protein",
                }
            ]
        }

    audit = []
    result = verify_recommendations([paper], _DeterministicClient(related), {}, audit=audit)
    assert result == [paper]
    assert audit[0]["related"] is True
    assert audit[0]["reason"] == "Both materials directly address dietary timing."
    assert "anchor_abstract" not in audit[0]
    assert paper.recommendation_matches == [
        {
            "anchor_title": "Meal timing and metabolic health",
            "anchor_url": "https://www.zotero.org/items/A1",
            "anchor_key": "A1",
            "similarity": 0.91,
            "reason": "Both materials directly address dietary timing.",
            "candidate_quote": "dietary protein timing",
            "anchor_quote": "meal timing and dietary protein",
        }
    ]
    assert "anchor_abstract" not in paper.recommendation_matches[0]


def test_verify_recommendations_retries_once_for_malformed_response():
    paper = _paper_with_anchor()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _chat_response({"recommendations": []})
        return _chat_response(
            {
                "recommendations": [
                    {
                        "id": "0",
                        "related": False,
                        "anchor_index": None,
                        "reason": "The supplied materials do not establish a direct relation.",
                        "candidate_quote": None,
                        "anchor_quote": None,
                    }
                ]
            }
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert verify_recommendations([paper], client, {}) == []
    assert len(calls) == 2


def _query_corpus():
    return [
        CorpusPaper(
            title="Heart failure exercise capacity",
            abstract="The paper evaluates exercise capacity in heart failure.",
            added_date=datetime(2026, 1, 1),
            paths=["topics/heart"],
        ),
        CorpusPaper(
            title="Dietary protein meal timing",
            abstract="The study discusses inflammatory signaling and meal timing.",
            added_date=datetime(2026, 1, 2),
            paths=["topics/nutrition"],
        ),
    ]


def _valid_query_payload(ids):
    return {
        "queries": [
            {"id": ids[1], "terms": ["meal timing", "inflammatory signaling"]},
            {"id": ids[0], "terms": ["heart failure", "exercise capacity"]},
        ]
    }


def test_build_zotero_queries_llm_validates_all_source_ids_and_order():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        ids = re.findall(r"Source ID: (\d+)", kwargs["messages"][1]["content"])
        return _chat_response(_valid_query_payload(ids))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    first = build_zotero_queries(_query_corpus(), openai_client=client, llm_params={})
    second = build_zotero_queries(_query_corpus(), openai_client=client, llm_params={})
    assert first == second
    assert any("heart failure" in query for query in first)
    assert any("meal timing" in query for query in first)
    assert len(calls) == 2


def test_build_zotero_queries_llm_retries_missing_id_once():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _chat_response({"queries": [{"id": "0", "terms": ["heart failure"]}]})
        return _chat_response(_valid_query_payload(["0", "1"]))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert build_zotero_queries(_query_corpus(), openai_client=client, llm_params={})
    assert len(calls) == 2
    assert "Correction:" in calls[1]["messages"][1]["content"]


def test_build_zotero_queries_llm_rejects_invented_terms_after_retry():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _chat_response(
            {
                "queries": [
                    {"id": "0", "terms": ["invented disease"]},
                    {"id": "1", "terms": ["meal timing"]},
                ]
            }
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RecommendationQueryError, match="failed closed"):
        build_zotero_queries(_query_corpus(), openai_client=client, llm_params={})
    assert len(calls) == 2


def test_build_zotero_queries_discards_ungrounded_extra_term_but_keeps_each_source():
    def create(**kwargs):
        return _chat_response({"queries": [
            {"id": 0, "terms": ["heart failure", "invented disease"]},
            {"id": 1, "terms": ["meal timing", "cohort"]},
        ]})

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    queries = build_zotero_queries(_query_corpus(), openai_client=client, llm_params={})
    assert any("heart failure" in query for query in queries)
    assert any("meal timing" in query for query in queries)
    assert not any("invented" in query or "cohort" in query for query in queries)


def test_request_failure_isolated_and_audited():
    papers = [_paper_with_anchor(), _paper_with_anchor()]
    client = _DeterministicClient(lambda ids, _: {"recommendations": [{"id": i, "related": True, "reason": "Direct relationship.", "anchor_index": 0, "candidate_quote": "dietary protein timing", "anchor_quote": "meal timing and dietary protein"} for i in ids]})
    original = client.create
    def create(**kwargs):
        if not client.calls:
            client.calls.append(kwargs)
            raise TimeoutError("secret https://example.test")
        return original(**kwargs)
    client.chat.completions.create = create
    audit = []
    result = verify_recommendations(papers, client, {}, batch_size=1, audit=audit)
    assert len(result) == 1 and result[0] is papers[1]
    assert not papers[0].recommendation_matches
    assert any(item.get("status") == "verification_unavailable" and item["related"] is None for item in audit)


def test_split_batch_recovers_and_has_bounded_requests():
    papers = [_paper_with_anchor() for _ in range(10)]
    client = _DeterministicClient(lambda ids, _: {"recommendations": [{"id": i, "related": False, "reason": "Unrelated topic.", "anchor_index": None, "candidate_quote": None, "anchor_quote": None} for i in ids]})
    original = client.create
    def create(**kwargs):
        ids = re.findall(r"Candidate ID: (\d+)", kwargs["messages"][1]["content"])
        if len(ids) > 1:
            client.calls.append(kwargs)
            raise TimeoutError("secret")
        return original(**kwargs)
    client.chat.completions.create = create
    verify_recommendations(papers, client, {}, batch_size=2)
    assert len(client.calls) <= 11


@pytest.mark.parametrize("status", [401, 429, 503])
def test_all_request_failures_safe_and_bounded(status):
    class Failure(Exception):
        status_code = status
    client = _DeterministicClient(lambda ids, _: None)
    def create(**kwargs):
        client.calls.append(kwargs)
        raise Failure("secret https://example.test/private")
    client.chat.completions.create = create
    with pytest.raises(RecommendationVerificationError) as raised:
        verify_recommendations([_paper_with_anchor() for _ in range(20)], client, {}, batch_size=2)
    assert str(status) in str(raised.value)
    assert "secret" not in str(raised.value) and "example.test" not in str(raised.value)
    assert len(client.calls) == (1 if status in (401, 429) else 7)


def test_invalid_quote_isolates_later_batch():
    papers = [_paper_with_anchor(), _paper_with_anchor()]
    calls = []
    def payload(ids, _):
        calls.append(ids)
        return {"recommendations": [{"id": ids[0], "related": True, "anchor_index": 0,
            "reason": "Direct relation.", "candidate_quote": "dietary protein timing" if len(calls) == 1 else "invented phrase", "anchor_quote": "meal timing and dietary protein"}]}
    audit = []
    result = verify_recommendations(papers, _DeterministicClient(payload), {}, batch_size=1, audit=audit)
    assert len(result) == 1 and result[0] is papers[0]
    assert not papers[1].recommendation_matches
    assert any(row.get("status") == "verification_unavailable" for row in audit)


def test_correction_502_isolates_batch_and_keeps_valid_result():
    papers = [_paper_with_anchor(), _paper_with_anchor()]
    client = _DeterministicClient(lambda ids, _: {"recommendations": [{"id": ids[0], "related": True, "reason": "Direct relationship.", "anchor_index": 0, "candidate_quote": "dietary protein timing", "anchor_quote": "meal timing and dietary protein"}]})
    original = client.create
    class Failure(Exception):
        status_code = 502
    def create(**kwargs):
        if len(client.calls) == 1:
            client.calls.append(kwargs)
            return _chat_response({"invalid": []})
        if len(client.calls) >= 2:
            client.calls.append(kwargs)
            raise Failure("secret")
        return original(**kwargs)
    client.chat.completions.create = create
    audit = []
    result = verify_recommendations(papers, client, {}, batch_size=1, audit=audit)
    assert len(result) == 1 and result[0] is papers[0]
    assert not papers[1].recommendation_matches
    assert any(row.get("status") == "verification_unavailable" for row in audit)


def test_three_split_recoveries_process_every_candidate():
    papers = [_paper_with_anchor() for _ in range(6)]
    client = _DeterministicClient(lambda ids, _: {"recommendations": [{"id": i, "related": True, "reason": "Direct relationship.", "anchor_index": 0, "candidate_quote": "dietary protein timing", "anchor_quote": "meal timing and dietary protein"} for i in ids]})
    original = client.create
    def create(**kwargs):
        ids = re.findall(r"Candidate ID: (\d+)", kwargs["messages"][1]["content"])
        if len(ids) > 1:
            client.calls.append(kwargs)
            raise TimeoutError("secret")
        return original(**kwargs)
    client.chat.completions.create = create
    result = verify_recommendations(papers, client, {}, batch_size=2)
    assert len(result) == 6 and all(a is b for a, b in zip(result, papers))
    assert len(client.calls) == 9


def test_fatal_failure_after_success_does_not_mutate():
    papers = [_paper_with_anchor(), _paper_with_anchor()]
    original_matches = [list(p.recommendation_matches) for p in papers]
    client = _DeterministicClient(lambda ids, _: {"recommendations": [{"id": ids[0], "related": True, "reason": "Direct relationship.", "anchor_index": 0, "candidate_quote": "dietary protein timing", "anchor_quote": "meal timing and dietary protein"}]})
    original = client.create
    class Failure(Exception):
        status_code = 401
    def create(**kwargs):
        if client.calls:
            raise Failure("secret")
        return original(**kwargs)
    client.chat.completions.create = create
    with pytest.raises(RecommendationVerificationError):
        verify_recommendations(papers, client, {}, batch_size=1)
    assert [p.recommendation_matches for p in papers] == original_matches


def test_correction_401_preserves_safe_status_and_paper():
    paper = _paper_with_anchor()
    original = list(paper.recommendation_matches)
    client = _DeterministicClient(lambda ids, _: {"invalid": []})
    first = client.create
    class Failure(Exception):
        status_code = 401
    def create(**kwargs):
        if client.calls:
            client.calls.append(kwargs)
            raise Failure("secret https://example.test")
        return first(**kwargs)
    client.chat.completions.create = create
    with pytest.raises(RecommendationVerificationError) as raised:
        verify_recommendations([paper], client, {})
    assert "401" in str(raised.value)
    assert "secret" not in str(raised.value) and "example.test" not in str(raised.value)
    assert len(client.calls) == 2
    assert paper.recommendation_matches == original
