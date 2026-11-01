"""Small, fail-safe client for the EasyScholar journal-rank endpoint."""

from __future__ import annotations

import math
import unicodedata
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Any, Mapping

import requests
from loguru import logger


_ENDPOINT = "https://www.easyscholar.cc/open/getPublicationRank"
_SOURCE = "https://www.easyscholar.cc/console/query"


def _normalise_title(value: Any) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(value.split())


def _clean_title(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    return " ".join(value.split())


def _finite_jif(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _raw_label(value: Any) -> str | None:
    """Keep EasyScholar's category label unchanged when it is a string."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _safe_numeric(value: Any) -> int | float | str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and value.isdigit():
        return value
    return None


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class EasyScholarClient:
    """Look up journal metrics with per-title caching and conservative failure handling."""

    def __init__(self, api_key: str, timeout: float = 10, min_interval: float = 0.55):
        self.api_key = str(api_key or "").strip()
        self.timeout = float(timeout)
        # Keep the production safety floor even if a caller passes a smaller value.
        self.min_interval = max(float(min_interval), 0.55)
        self._last_request_at: float | None = None
        self._cache: dict[str, dict[str, Any] | None] = {}
        self._disabled = False

    def _pace(self) -> None:
        now = monotonic()
        if self._last_request_at is not None:
            wait_for = self.min_interval - (now - self._last_request_at)
            if wait_for > 0:
                sleep(wait_for)
                now += wait_for
        self._last_request_at = now

    @staticmethod
    def _warn_request_failure() -> None:
        # Do not include the URL, title, exception text, request parameters, or key.
        logger.warning("EasyScholar request failed")

    def _parse(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, Mapping):
            logger.warning("EasyScholar response was not an object")
            return None

        code = payload.get("code")
        if code != 200 and code != "200":
            safe_code = _safe_numeric(code)
            if safe_code is None:
                logger.warning("EasyScholar response code was unsuccessful")
            else:
                logger.warning("EasyScholar response code {}", safe_code)
            # These two credential errors are documented or directly verified.
            # A missing journal or an unknown business error must not prevent
            # other titles from being queried.
            self._disabled = code in (40002, "40002", 40005, "40005")
            return None

        data = payload.get("data")
        official_rank = data.get("officialRank") if isinstance(data, Mapping) else None
        rank = official_rank.get("all") if isinstance(official_rank, Mapping) else None
        if not isinstance(rank, Mapping):
            logger.warning("EasyScholar response did not contain official rankings")
            return None

        metrics: dict[str, Any] = {
            "metrics_provider": "EasyScholar",
            "queried_at": datetime.now(timezone.utc).date().isoformat(),
        }
        recognised = False

        jif = _finite_jif(rank.get("sciif"))
        if jif is not None:
            recognised = True
            metrics["jif"] = jif
            metrics["jif_year"] = None
            metrics["source"] = _SOURCE

        jcr_quartile = _raw_label(rank.get("sci"))
        if jcr_quartile is not None:
            recognised = True
            metrics["jcr_quartile"] = jcr_quartile
            metrics["jcr_category"] = None
            metrics["jcr_year"] = None
            metrics["jcr_source"] = _SOURCE

        cas_quartile: dict[str, str] = {}
        cas_big = _raw_label(rank.get("sciUp"))
        cas_small = _raw_label(rank.get("sciUpSmall"))
        if cas_big is not None:
            cas_quartile["big"] = cas_big
        if cas_small is not None:
            cas_quartile["small"] = cas_small
        if cas_quartile:
            recognised = True
            metrics["cas_quartile"] = cas_quartile
            metrics["cas_category"] = None
            metrics["cas_version"] = "升级版"
            metrics["cas_year"] = None
            metrics["cas_source"] = _SOURCE

        return metrics if recognised else None

    def lookup(self, journal: str) -> dict[str, Any] | None:
        """Return parsed metrics for ``journal`` or ``None`` when unavailable."""
        cache_key = _normalise_title(journal)
        if not cache_key:
            return None
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.api_key or self._disabled:
            self._cache[cache_key] = None
            return None

        title = _clean_title(journal)
        self._pace()
        try:
            response = requests.get(
                _ENDPOINT,
                params={"secretKey": self.api_key, "publicationName": title},
                timeout=self.timeout,
                allow_redirects=False,
            )
        except Exception:
            self._warn_request_failure()
            self._cache[cache_key] = None
            return None

        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            safe_status = _safe_numeric(status_code)
            if safe_status is None:
                logger.warning("EasyScholar HTTP request failed")
            else:
                logger.warning("EasyScholar HTTP status {}", safe_status)
            self._disabled = status_code in (401, 403, 429)
            self._cache[cache_key] = None
            return None

        try:
            payload = response.json()
        except Exception:
            self._warn_request_failure()
            self._cache[cache_key] = None
            return None

        metrics = self._parse(payload)
        self._cache[cache_key] = metrics
        return metrics


def enrich_journal_metrics(papers: list[Any], journal_config: Any) -> list[Any]:
    """Enrich papers and, when enabled, apply the configured finite-JIF gate.

    ``papers`` deliberately stays duck-typed so this helper does not import the
    heavier paper protocol or any retrieval dependencies.
    """
    easyscholar = _config_value(journal_config, "easyscholar", {})
    if not _config_value(easyscholar, "enabled", False):
        return papers

    api_key = _config_value(easyscholar, "api_key", "")
    if not str(api_key or "").strip():
        return papers

    client = EasyScholarClient(
        api_key,
        timeout=float(_config_value(easyscholar, "timeout", 10)),
        min_interval=float(_config_value(easyscholar, "min_interval", 0.55)),
    )

    min_jif = _finite_jif(_config_value(journal_config, "min_jif", 0))
    if min_jif is None:
        raise ValueError("journals.min_jif must be a finite non-negative number")

    retained: list[Any] = []
    enriched_count = 0
    for paper in papers:
        journal = getattr(paper, "journal", None)
        original_metrics = getattr(paper, "journal_metrics", None)
        static_metrics = dict(original_metrics) if isinstance(original_metrics, Mapping) else {}

        live_metrics = client.lookup(journal) if journal else None
        final_metrics = static_metrics
        if live_metrics is not None:
            final_metrics.update(live_metrics)
            paper.journal_metrics = final_metrics
            enriched_count += 1

        if _config_value(journal_config, "enabled", False):
            jif = _finite_jif(final_metrics.get("jif"))
            if jif is None or jif < min_jif:
                continue
        retained.append(paper)

    logger.info(
        "EasyScholar metadata: enriched {} paper(s); IF filter excluded {} paper(s)",
        enriched_count, len(papers) - len(retained),
    )
    return retained
