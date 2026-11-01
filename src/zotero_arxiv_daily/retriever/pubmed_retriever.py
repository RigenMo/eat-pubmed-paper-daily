import re
import requests
import unicodedata
from time import sleep
from datetime import datetime, timedelta, timezone
from typing import Any
from loguru import logger
from .base import BaseRetriever, register_retriever
from ..protocol import Paper

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_RATE_SLEEP = 0.34  # ≤3 requests/second without API key


def _parse_affiliation(ad_str: str) -> str:
    """Extract a clean institution name from a MEDLINE AD field.

    MEDLINE AD fields look like:
      "Department of Chemistry, MIT, Cambridge, MA 02139, USA. Electronic address: x@x.edu."
    We take at most the first two comma-separated components that are not
    postal codes, strip trailing noise, and join them.
    """
    # Remove "Electronic address: ..." suffix (common in modern MEDLINE)
    ad_str = re.sub(r"\.\s*Electronic\s+address:.*$", "", ad_str, flags=re.IGNORECASE).strip()
    parts = [p.strip().rstrip(".") for p in ad_str.split(",")]
    result = []
    for part in parts:
        if not part:
            continue
        if re.match(r"^\d{2,}", part):   # starts with digits → postal code, stop here
            break
        result.append(part)
        if len(result) == 2:             # keep department + institution (2 levels max)
            break
    return ", ".join(result) if result else (parts[0] if parts else ad_str[:120])


def _normalize_journal_name(value: Any) -> str:
    """Normalize a journal title for exact allowlist matching.

    Matching is case-insensitive and tolerant of punctuation/whitespace
    differences, but it is not substring matching: the resulting normalized
    title must be one of the configured names or aliases.
    """
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _ncbi_get(url: str, params: dict[str, Any], retries: int = 5) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == retries - 1:
                raise
            logger.warning(f"NCBI request failed ({exc}), retry {attempt + 1}/{retries}")
            sleep(10)


def _parse_medline(medline_text: str) -> list[dict[str, Any]]:
    """Parse PubMed MEDLINE flat-file format into a list of record dicts."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_tag = ""

    for line in medline_text.splitlines():
        if len(line) < 4:
            continue
        tag = line[:4].strip()
        value = line[6:].strip() if len(line) > 6 else ""

        if tag:
            current_tag = tag
            if tag == "PMID":
                if current:
                    records.append(current)
                current = {"PMID": value}
            elif tag in current:
                # multi-line / multi-value fields: collect as list
                if isinstance(current[tag], list):
                    current[tag].append(value)
                else:
                    current[tag] = [current[tag], value]
            else:
                current[tag] = value
        else:
            # continuation line
            if current_tag in current:
                if isinstance(current[current_tag], list):
                    current[current_tag][-1] += " " + value
                else:
                    current[current_tag] += " " + value

    if current:
        records.append(current)
    return records


@register_retriever("pubmed")
class PubMedRetriever(BaseRetriever):
    name = "pubmed"

    # Minimum result count before escalating to the next fallback level.
    _MIN_RESULTS = 15

    def __init__(self, config):
        # BaseRetriever.__init__ calls getattr(config.source, self.name)
        # so config.source.pubmed must exist
        super().__init__(config)
        self._pubmed_query: str | None = None        # strict query (1-2 AND groups)
        self._pubmed_broad_query: str | None = None  # broad query (single OR group)
        self._max_results: int = int(self.retriever_config.get("max_results", 200))

        # Journal filtering is deliberately fail-closed. A configured but
        # empty allowlist must never turn into an unrestricted PubMed query.
        journal_config = self.config.get("journals", {})
        self._journal_filter_enabled = bool(journal_config.get("enabled", False))
        self._allowed_journals: set[str] = set()
        self._journal_query_names: list[str] = []
        if self._journal_filter_enabled:
            min_jif = float(journal_config.get("min_jif", 0))
            seen_query_names: set[str] = set()
            for entry in journal_config.get("whitelist", []) or []:
                try:
                    jif = float(entry.get("jif", 0))
                except (TypeError, ValueError):
                    logger.warning(f"Ignoring journal with invalid JIF: {entry!r}")
                    continue
                if jif < min_jif:
                    continue

                name = str(entry.get("name", "")).strip()
                aliases = entry.get("aliases", []) or []
                if isinstance(aliases, str):
                    aliases = [aliases]
                names = [name, *[str(alias).strip() for alias in aliases]]
                names = [candidate for candidate in names if candidate]
                for candidate in names:
                    normalized = _normalize_journal_name(candidate)
                    if normalized:
                        self._allowed_journals.add(normalized)
                    if candidate and candidate.casefold() not in seen_query_names:
                        seen_query_names.add(candidate.casefold())
                        self._journal_query_names.append(candidate)

            if self._journal_query_names:
                self._journal_constraint = " OR ".join(
                    f'"{name.replace(chr(34), "")}"[Journal]'
                    for name in self._journal_query_names
                )
            else:
                self._journal_constraint = None
                logger.warning(
                    "PubMed journal filter is enabled but its shortlist is empty; "
                    "all PubMed queries will fail closed."
                )
        else:
            self._journal_constraint = None

    def set_pubmed_query(self, pubmed_query: str) -> None:
        """Inject a structured PubMed strict query (1-2 AND concept groups)."""
        self._pubmed_query = pubmed_query or None

    def set_pubmed_broad_query(self, pubmed_broad_query: str) -> None:
        """Inject a broad PubMed query (single OR concept group, no AND)."""
        self._pubmed_broad_query = pubmed_broad_query or None

    def set_max_results(self, n: int) -> None:
        """Called by Executor to apply mix_mode quota."""
        self._max_results = n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _date_filter(self) -> str:
        retrieval_days = int(self.config.executor.get("retrieval_days", 1))
        start = (datetime.now(timezone.utc) - timedelta(days=retrieval_days)).strftime("%Y/%m/%d")
        return f'("{start}"[PDAT] : "3000"[PDAT])'

    def _fetch_pmids(self, query: str) -> list[str]:
        if self._journal_filter_enabled:
            if not self._journal_constraint:
                return []
            query = f"({query}) AND ({self._journal_constraint})"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self._max_results,
            "retmode": "json",
            "sort": "pub_date",
        }
        sleep(_RATE_SLEEP)
        resp = _ncbi_get(_ESEARCH_URL, params)
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids

    def _fetch_details(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []
        all_records: list[dict[str, Any]] = []
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "medline",
                "retmode": "text",
            }
            sleep(_RATE_SLEEP)
            resp = _ncbi_get(_EFETCH_URL, params)
            records = _parse_medline(resp.text)
            all_records.extend(records)
        return all_records

    def is_allowed_journal(self, journal: Any) -> bool:
        """Return whether a final paper journal passes the exact allowlist gate."""
        if not self._journal_filter_enabled:
            return True
        return _normalize_journal_name(journal) in self._allowed_journals

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        """Progressive fallback across query levels, escalating only when results < _MIN_RESULTS."""
        date_filter = self._date_filter()
        min_n = self._MIN_RESULTS
        seen_queries: set[str] = set()

        def _try(label: str, kw: str) -> list[str]:
            q = f"({kw}) AND {date_filter}"
            if q in seen_queries:
                return []
            seen_queries.add(q)
            logger.info(f"PubMed [{label}]: {q}")
            pmids = self._fetch_pmids(q)
            logger.info(f"PubMed [{label}] → {len(pmids)} result(s)")
            return pmids

        def _strip_tiab(q: str) -> str:
            return re.sub(r"\[tiab\]", "", q).strip()

        pmids: list[str] = []

        # Mode 5 deliberately has no keyword query: retrieve the recent PubMed
        # window first, then let the Zotero-based reranker select relevant papers.
        # Keyword modes must not silently broaden to an unfiltered date-only
        # search when keyword extraction failed to provide either query.
        if not self._pubmed_query and not self._pubmed_broad_query:
            if int(self.config.search.mode) < 5:
                raise ValueError(
                    "PubMed query is required when search.mode < 5; "
                    "provide keywords or fix keyword extraction."
                )

            logger.info(f"PubMed [date-only]: {date_filter}")
            pmids = self._fetch_pmids(date_filter)
            logger.info(f"PubMed [date-only] → {len(pmids)} result(s)")
            if not pmids:
                logger.warning("PubMed: no results found for the date-only query")
                return []
            return self._finalise(pmids)

        # ── Level 1: strict query (1-2 AND concept groups, [tiab]) ──────────
        if self._pubmed_query:
            pmids = _try("strict", self._pubmed_query)
            if len(pmids) >= min_n:
                return self._finalise(pmids)

        # ── Level 2: strict without [tiab] → full-field search ──────────────
        if self._pubmed_query and "[tiab]" in self._pubmed_query:
            pmids_fb = _try("strict|no-tiab", _strip_tiab(self._pubmed_query))
            if len(pmids_fb) >= min_n:
                return self._finalise(pmids_fb)
            if len(pmids_fb) > len(pmids):
                pmids = pmids_fb   # keep the better count for later

        # ── Level 3: broad query (single OR group, [tiab]) ──────────────────
        if self._pubmed_broad_query:
            pmids_fb = _try("broad", self._pubmed_broad_query)
            if len(pmids_fb) >= min_n:
                return self._finalise(pmids_fb)
            if len(pmids_fb) > len(pmids):
                pmids = pmids_fb

        # ── Level 4: broad without [tiab] ───────────────────────────────────
        if self._pubmed_broad_query and "[tiab]" in self._pubmed_broad_query:
            pmids_fb = _try("broad|no-tiab", _strip_tiab(self._pubmed_broad_query))
            if len(pmids_fb) > len(pmids):
                pmids = pmids_fb

        if not pmids:
            logger.warning("PubMed: no results found after all fallback levels")
            return []

        logger.info(f"PubMed: using {len(pmids)} PMIDs (best fallback level)")
        return self._finalise(pmids)

    def _finalise(self, pmids: list[str]) -> list[dict[str, Any]]:
        logger.info(f"PubMed: fetching details for {len(pmids)} PMIDs...")
        records = self._fetch_details(pmids)
        if not self._journal_filter_enabled:
            return records

        kept: list[dict[str, Any]] = []
        excluded = 0
        for record in records:
            candidates = []
            for field in ("JT", "TA"):
                value = record.get(field)
                if isinstance(value, list):
                    candidates.extend(value)
                elif value:
                    candidates.append(value)
            if any(self.is_allowed_journal(value) for value in candidates):
                kept.append(record)
            else:
                excluded += 1

        logger.info(
            f"PubMed journal filter: kept {len(kept)} record(s), "
            f"excluded {excluded} unknown/unlisted record(s)."
        )
        return kept

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        pmid = raw_paper.get("PMID", "")
        title = raw_paper.get("TI", "")
        if not title:
            return None

        # Abstract: AB field, may be a list for structured abstracts
        ab = raw_paper.get("AB", "")
        if isinstance(ab, list):
            ab = "\n\n".join(ab)
        if not ab:
            return None  # skip papers with no abstract

        # Authors: AU field
        au = raw_paper.get("AU", [])
        if isinstance(au, str):
            au = [au]
        authors = [a.strip() for a in au]

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        # Journal title (full name from JT, fallback to abbreviation TA)
        journal = raw_paper.get("JT") or raw_paper.get("TA") or None

        # DOI → build PDF hint (not always available; leave pdf_url as None)
        lid = raw_paper.get("LID", "")
        if isinstance(lid, list):
            lid = " ".join(lid)
        doi = None
        for part in lid.split():
            if part.startswith("10."):
                doi = part
                break
        pdf_url = f"https://doi.org/{doi}" if doi else None

        # Affiliations: parse MEDLINE AD field directly (no LLM needed for PubMed)
        ad = raw_paper.get("AD", [])
        if isinstance(ad, str):
            ad = [ad]
        if ad:
            cleaned = [_parse_affiliation(a) for a in ad if a.strip()]
            # Deduplicate while preserving order
            seen: set[str] = set()
            affiliations: list[str] = []
            for a in cleaned:
                if a not in seen:
                    seen.add(a)
                    affiliations.append(a)
        else:
            affiliations = None

        journal_metrics = None
        for entry in self.config.get("journals", {}).get("whitelist", []) or []:
            aliases = entry.get("aliases", []) or []
            if isinstance(aliases, str):
                aliases = [aliases]
            names = [entry.get("name", ""), *aliases]
            normalized_names = {_normalize_journal_name(name) for name in names if name}
            if journal and _normalize_journal_name(journal) in normalized_names:
                journal_metrics = dict(entry)
                break

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=ab,
            url=url,
            pdf_url=pdf_url,
            full_text=None,
            journal=journal,
            journal_metrics=journal_metrics,
            affiliations=affiliations,
        )
