from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type

# Maps search.mode (1-5) to keyword weight alpha
_MODE_TO_ALPHA = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0}


class BaseReranker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    def rerank(
        self,
        candidates: list[Paper],
        corpus: list[CorpusPaper],
        keyword_query: str | None = None,
        favorite_journals: list[str] | None = None,
    ) -> list[Paper]:
        search_mode = int(self.config.search.mode)
        alpha = _MODE_TO_ALPHA.get(search_mode, 0.0)

        keyword_scores = np.zeros(len(candidates))
        corpus_scores = np.zeros(len(candidates))

        # Build text representations: title + abstract for richer semantic matching
        candidate_texts = [f"{c.title}. {c.abstract}" for c in candidates]

        # A reranker can be reused by the executor.  Do not leave anchors from
        # an earlier run on a Paper when this run has no corpus evidence.
        for candidate in candidates:
            candidate.recommendation_matches = []

        # --- keyword scoring ---
        if candidates and alpha > 0 and keyword_query:
            kw_sim = self.get_similarity_score(
                candidate_texts, [keyword_query]
            )  # shape: (n_candidates, 1)
            keyword_scores = kw_sim[:, 0]  # (n_candidates,)

        # --- corpus (Zotero) scoring ---
        if candidates and alpha < 1.0 and len(corpus) > 0:
            # Keep the caller's order: the similarity matrix columns refer to
            # these exact source records.  A global time-decayed average lets
            # dozens of unrelated Zotero papers dilute the nearest topic.  A
            # candidate is instead scored by its closest three (or fewer)
            # source papers, which also gives the verifier concrete anchors.
            corpus_records = list(corpus)
            corpus_texts = [
                f"{getattr(c, 'title', '')}. {getattr(c, 'abstract', '')}"
                for c in corpus_records
            ]
            sim = self.get_similarity_score(
                candidate_texts,
                corpus_texts,
            )  # shape: (n_candidates, n_corpus)
            expected_shape = (len(candidates), len(corpus_records))
            if sim.shape != expected_shape:
                raise ValueError(
                    "Reranker similarity shape does not match candidate and corpus counts"
                )

            top_k = min(3, len(corpus_records))
            for candidate_index, row in enumerate(sim):
                # Stable sorting preserves the source order for exact ties.
                nearest_indices = np.argsort(-row, kind="stable")[:top_k]
                nearest_similarities = row[nearest_indices]
                corpus_scores[candidate_index] = float(np.mean(nearest_similarities))

                matches = []
                for source_index in nearest_indices:
                    source = corpus_records[int(source_index)]
                    matches.append(
                        {
                            "anchor_title": str(getattr(source, "title", "") or ""),
                            "anchor_url": str(getattr(source, "url", "") or ""),
                            "anchor_key": str(getattr(source, "key", "") or ""),
                            "anchor_abstract": str(
                                getattr(source, "abstract", "") or ""
                            ),
                            "similarity": float(row[int(source_index)]),
                        }
                    )
                candidates[candidate_index].recommendation_matches = matches

        scores = (alpha * keyword_scores + (1.0 - alpha) * corpus_scores) * 10

        # --- journal bonus: boost papers from user's favorite journals ---
        if favorite_journals:
            fav_set = {j.lower().strip() for j in favorite_journals}
            for i, c in enumerate(candidates):
                if c.journal and c.journal.lower().strip() in fav_set:
                    scores[i] *= 1.5

        # Similarity implementations are expected to return values on a
        # cosine-like 0..1 scale.  Keep the public score a 0..10 display score
        # even if a provider returns a slightly out-of-range value or the
        # journal bonus is applied.
        scores = np.clip(scores, 0.0, 10.0)

        for s, c in zip(scores, candidates):
            c.score = float(s)
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return candidates

    @abstractmethod
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
