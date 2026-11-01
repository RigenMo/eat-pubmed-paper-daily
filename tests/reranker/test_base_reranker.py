"""Tests for BaseReranker: scoring, nearest anchors, and registration."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray):
        # search.mode=5 -> alpha=0.0 -> pure Zotero-corpus scoring, matching
        # what these tests exercise (no keyword_query is passed to rerank()).
        self.config = OmegaConf.create({"search": {"mode": 5}})
        self._sim = sim_matrix

    def get_similarity_score(self, s1, s2):
        return self._sim


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_uses_closest_three_without_dilution():
    corpus = make_sample_corpus(5)
    papers = [make_sample_paper(title="P")]

    # Three close anchors should determine the score; two unrelated records
    # must not dilute it as a global corpus average would.
    sim = np.array([[0.9, 0.8, 0.7, 0.0, 0.0]])
    ranked = StubReranker(sim).rerank(papers, corpus)
    assert ranked[0].score == pytest.approx(8.0)
    assert [match["anchor_title"] for match in ranked[0].recommendation_matches] == [
        "Corpus Paper 0",
        "Corpus Paper 1",
        "Corpus Paper 2",
    ]


def test_rerank_anchor_fields_use_source_defaults():
    corpus = make_sample_corpus(1)
    corpus[0].key = "ZOTKEY"
    corpus[0].url = "https://www.zotero.org/items/ZOTKEY"
    papers = [make_sample_paper(title="P")]

    ranked = StubReranker(np.array([[0.5]])).rerank(papers, corpus)
    assert ranked[0].recommendation_matches == [
        {
            "anchor_title": "Corpus Paper 0",
            "anchor_url": "https://www.zotero.org/items/ZOTKEY",
            "anchor_key": "ZOTKEY",
            "anchor_abstract": "Abstract for corpus paper 0.",
            "similarity": 0.5,
        }
    ]


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")
