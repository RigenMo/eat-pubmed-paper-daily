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

    result = verify_recommendations([paper], _DeterministicClient(related), {})
    assert result == [paper]
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
