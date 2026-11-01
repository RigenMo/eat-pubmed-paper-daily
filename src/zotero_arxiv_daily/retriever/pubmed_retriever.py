import re
import calendar
import requests
import unicodedata
import xml.etree.ElementTree as ET
from time import sleep
from datetime import datetime, timedelta, timezone
from typing import Any
from loguru import logger
from .base import BaseRetriever, register_retriever
from ..protocol import Paper

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_RATE_SLEEP = 0.34  # ≤3 requests/second without API key

_MONTH_NUMBERS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

_HISTORY_STATUSES = {
    "received",
    "revised",
    "accepted",
    "aheadofprint",
    "epublish",
    "ppublish",
    "ecollection",
    # These are PubMed processing/indexing events, not peer-review events.
    "entrez",
    "pubmed",
    "medline",
    "pmc-release",
}


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


def _xml_local_name(tag: str) -> str:
    """Return an XML tag name without an optional namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def _xml_child(parent: ET.Element | None, name: str) -> ET.Element | None:
    if parent is None:
        return None
    for child in list(parent):
        if _xml_local_name(child.tag) == name:
            return child
    return None


def _xml_children(parent: ET.Element | None, name: str) -> list[ET.Element]:
    if parent is None:
        return []
    return [child for child in list(parent) if _xml_local_name(child.tag) == name]


def _xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _normalise_pubmed_month(value: str) -> str | None:
    value = value.strip()
    if value.isdigit():
        month = int(value)
        return f"{month:02d}" if 1 <= month <= 12 else None
    return _MONTH_NUMBERS.get(value.casefold())


def _parse_pubmed_date(node: ET.Element | None) -> str | None:
    """Render a PubMed ``PubDate``/``ArticleDate`` without inventing precision.

    PubMed permits year-only, year-month, year-month-day, and season dates.
    Missing month/day values therefore remain missing rather than being
    converted to January or day one.  A ``MedlineDate`` fallback is retained
    verbatim because older citations can use free-form issue dates.
    """
    if node is None:
        return None

    year = _xml_text(_xml_child(node, "Year"))
    month_raw = _xml_text(_xml_child(node, "Month"))
    day = _xml_text(_xml_child(node, "Day"))
    season = _xml_text(_xml_child(node, "Season"))

    if year:
        month = _normalise_pubmed_month(month_raw) if month_raw else None
        # PubMed's structured year is normally four digits.  Keep a supplied
        # value if an old record uses a non-standard but still informative
        # year token; do not silently turn it into a datetime.
        if month:
            if day and re.fullmatch(r"\d{1,2}", day):
                return f"{year}-{month}-{int(day):02d}"
            return f"{year}-{month}"
        if month_raw:
            # Preserve ranges such as ``Oct-Dec`` that cannot be represented
            # as one exact ISO month.
            if day:
                return f"{year}-{month_raw}-{day}"
            return f"{year}-{month_raw}"
        if season:
            return f"{year} {season}"
        return year

    medline_date = _xml_text(_xml_child(node, "MedlineDate"))
    return medline_date or None


def _full_date_key(value: str) -> tuple[int, int, int] | None:
    """Return a sortable key only for a complete YYYY-MM-DD value."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.year, parsed.month, parsed.day


def _publication_in_window(value: str | None, start, today) -> bool:
    """Accept only dates whose entire stated precision fits the window."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        return False
    parts = [int(part) for part in value.split("-")]
    try:
        year = parts[0]
        month = parts[1] if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else 1
        first = datetime(year, month, day).date()
        last = first
        if len(parts) == 1:
            last = datetime(year, 12, 31).date()
        elif len(parts) == 2:
            last = datetime(year, month, calendar.monthrange(year, month)[1]).date()
    except ValueError:
        return False
    return start <= first and last <= today


def _pubmed_source(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""


def _choose_publication_date(
    article_electronic: str | None,
    article_print: str | None,
    history_events: list[tuple[str, str]],
    journal_date: str | None,
) -> str | None:
    """Choose the first actual publication date available in a citation.

    ``ArticleDate`` electronic is the strongest online-first signal.  If it
    is absent, explicit publication events from ``History`` are preferred to
    a later issue date.  When only the history has publication events, a
    complete date is compared; partial dates retain PubMed's event order and
    status priority instead of receiving a fabricated day.
    """
    if article_electronic:
        return article_electronic

    history_publication = [
        (status, date)
        for status, date in history_events
        if status in {"aheadofprint", "epublish", "ppublish"}
    ]
    if history_publication:
        complete = [(status, date, _full_date_key(date)) for status, date in history_publication]
        complete = [item for item in complete if item[2] is not None]
        if complete:
            return min(complete, key=lambda item: item[2])[1]
        # PubMed's order is meaningful when precision is incomplete.  Keep
        # the first supplied event rather than inventing chronology.
        return history_publication[0][1]

    if article_print:
        return article_print
    return journal_date


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed EFetch XML into the existing raw-paper shape.

    The parser intentionally uses only the stdlib XML parser and reads the
    citation, author, affiliation, DOI, abstract, journal-date, and PubMed
    history elements needed by the retriever.  PubMed processing statuses
    (``entrez``, ``pubmed``, ``medline``) are kept as events so downstream
    rendering can distinguish indexing from peer-review milestones.
    """
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, TypeError) as exc:
        raise ValueError("PubMed EFetch returned invalid XML") from exc

    records: list[dict[str, Any]] = []
    for pubmed_article in root.iter():
        if _xml_local_name(pubmed_article.tag) != "PubmedArticle":
            continue

        citation = _xml_child(pubmed_article, "MedlineCitation")
        article = _xml_child(citation, "Article")
        if citation is None or article is None:
            continue

        pmid = _xml_text(_xml_child(citation, "PMID"))
        record: dict[str, Any] = {"PMID": pmid}

        title = _xml_text(_xml_child(article, "ArticleTitle"))
        if title:
            record["TI"] = title

        abstract_node = _xml_child(article, "Abstract")
        abstract_parts: list[str] = []
        for abstract_text in _xml_children(abstract_node, "AbstractText"):
            value = _xml_text(abstract_text)
            if not value:
                continue
            label = abstract_text.attrib.get("Label", "").strip()
            if label:
                value = f"{label}: {value}"
            abstract_parts.append(value)
        if abstract_parts:
            record["AB"] = abstract_parts[0] if len(abstract_parts) == 1 else abstract_parts

        journal = _xml_child(article, "Journal")
        journal_title = _xml_text(_xml_child(journal, "Title"))
        journal_abbrev = _xml_text(_xml_child(journal, "ISOAbbreviation"))
        medline_info = _xml_child(citation, "MedlineJournalInfo")
        medline_abbrev = _xml_text(_xml_child(medline_info, "MedlineTA"))
        if journal_title:
            record["JT"] = journal_title
        if journal_abbrev or medline_abbrev:
            record["TA"] = journal_abbrev or medline_abbrev

        journal_issue = _xml_child(journal, "JournalIssue")
        journal_date = _parse_pubmed_date(_xml_child(journal_issue, "PubDate"))

        author_list = _xml_child(article, "AuthorList")
        authors: list[str] = []
        affiliations: list[str] = []
        for author in _xml_children(author_list, "Author"):
            collective = _xml_text(_xml_child(author, "CollectiveName"))
            last_name = _xml_text(_xml_child(author, "LastName"))
            initials = _xml_text(_xml_child(author, "Initials"))
            fore_name = _xml_text(_xml_child(author, "ForeName"))
            suffix = _xml_text(_xml_child(author, "Suffix"))
            if collective:
                name = collective
            else:
                name_parts = [part for part in (last_name, initials or fore_name, suffix) if part]
                name = " ".join(name_parts)
            if name:
                authors.append(name)
            for affiliation_info in _xml_children(author, "AffiliationInfo"):
                for affiliation in _xml_children(affiliation_info, "Affiliation"):
                    value = _xml_text(affiliation)
                    if value:
                        affiliations.append(value)
        if authors:
            record["AU"] = authors
        if affiliations:
            record["AD"] = affiliations

        doi_values: list[str] = []
        for elocation in _xml_children(article, "ELocationID"):
            if elocation.attrib.get("EIdType", "").casefold() == "doi":
                value = _xml_text(elocation)
                if value and value not in doi_values:
                    doi_values.append(value)

        pubmed_data = _xml_child(pubmed_article, "PubmedData")
        article_id_list = _xml_child(pubmed_data, "ArticleIdList")
        for article_id in _xml_children(article_id_list, "ArticleId"):
            if article_id.attrib.get("IdType", "").casefold() == "doi":
                value = _xml_text(article_id)
                if value and value not in doi_values:
                    doi_values.append(value)
        if doi_values:
            doi_fields = [f"{value} [doi]" for value in doi_values]
            record["LID"] = doi_fields[0] if len(doi_fields) == 1 else doi_fields

        article_electronic = None
        article_print = None
        for article_date in _xml_children(article, "ArticleDate"):
            date = _parse_pubmed_date(article_date)
            if not date:
                continue
            date_type = article_date.attrib.get("DateType", "").casefold()
            if date_type == "electronic" and article_electronic is None:
                article_electronic = date
            elif date_type == "print" and article_print is None:
                article_print = date

        history_events: list[tuple[str, str]] = []
        history = _xml_child(pubmed_data, "History")
        for pubmed_pub_date in _xml_children(history, "PubMedPubDate"):
            status = pubmed_pub_date.attrib.get("PubStatus", "").strip().casefold()
            date = _parse_pubmed_date(pubmed_pub_date)
            if status in _HISTORY_STATUSES and date:
                history_events.append((status, date))

        publication_date = _choose_publication_date(
            article_electronic=article_electronic,
            article_print=article_print,
            history_events=history_events,
            journal_date=journal_date,
        )

        source = _pubmed_source(pmid)
        publication_history = [
            {"event": status, "date": date, "source": source}
            for status, date in history_events
        ]
        if publication_date:
            publication_history.append(
                {"event": "publication", "date": publication_date, "source": source}
            )

        # Sort only when every returned date is a complete, valid date.  A
        # year-only/season/issue-range value has insufficient precision to
        # establish chronology, so preserve the source event order instead.
        date_keys = [_full_date_key(item["date"]) for item in publication_history]
        if publication_history and all(key is not None for key in date_keys):
            publication_history = [
                item
                for _, item in sorted(
                    zip(date_keys, publication_history), key=lambda pair: pair[0]
                )
            ]

        record["publication_date"] = publication_date
        record["publication_history"] = publication_history
        records.append(record)

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
        self._pubmed_queries: list[str] = []          # corpus/theme queries for mode 5
        self._pubmed_queries_configured = False
        self._max_results: int = int(self.retriever_config.get("max_results", 200))
        recommendation_config = config.get("recommendations", {})
        self._publication_days = int(config.executor.get("retrieval_days", 1))
        if recommendation_config.get("enabled", False) and int(config.search.mode) == 5:
            self._publication_days = int(recommendation_config.get("publication_days", 30))
        if self._publication_days < 1:
            raise ValueError("Publication window must be at least one day")

        # Journal filtering is deliberately fail-closed. A configured but
        # empty allowlist must never turn into an unrestricted PubMed query.
        journal_config = self.config.get("journals", {})
        self._dynamic_jif = bool(
            journal_config.get("enabled", False)
            and str(journal_config.get("mode", "")).casefold() == "easyscholar"
        )
        # With EasyScholar mode selected, journal selection happens after
        # enrichment/ranking.  Applying the static shortlist here would
        # prevent the dynamic source from seeing otherwise eligible journals.
        self._journal_filter_enabled = bool(journal_config.get("enabled", False)) and not self._dynamic_jif
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

    def set_pubmed_queries(self, queries: list[str] | None) -> None:
        """Inject corpus/theme queries used to build a balanced candidate pool.

        The configured flag is kept separately from the cleaned list: an
        explicitly empty corpus must produce no PubMed candidates rather than
        silently reverting to an unrestricted date-only search.
        """
        self._pubmed_queries_configured = True
        self._pubmed_queries = [
            query.strip()
            for query in (queries or [])
            if isinstance(query, str) and query.strip()
        ]

    def set_max_results(self, n: int) -> None:
        """Called by Executor to apply mix_mode quota."""
        self._max_results = n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _date_filter(self) -> str:
        """Search publication and electronic dates using explicit UTC bounds.

        https://pubmed.ncbi.nlm.nih.gov/help/#publication-date-dp
        """
        retrieval_days = self._publication_days
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=retrieval_days)
        lower = start.strftime("%Y/%m/%d")
        upper = today.strftime("%Y/%m/%d")
        return f'(("{lower}"[PDAT] : "{upper}"[PDAT]) OR ("{lower}"[EPDAT] : "{upper}"[EPDAT]))'

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

    def _fetch_pmids_with_limit(self, query: str, limit: int) -> list[str]:
        """Fetch one theme query without changing the public max-result quota.

        The temporary instance value keeps compatibility with tests and
        callers that monkeypatch ``_fetch_pmids(self, query)``.
        """
        original_max = self._max_results
        self._max_results = max(1, int(limit))
        try:
            return self._fetch_pmids(query)
        finally:
            self._max_results = original_max

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
                "rettype": "abstract",
                "retmode": "xml",
            }
            sleep(_RATE_SLEEP)
            resp = _ncbi_get(_EFETCH_URL, params)
            response_text = resp.text or ""
            # XML is the authoritative path for publication dates and
            # PubMedPubDate history.  Keep MEDLINE parsing for old fixtures
            # and any legacy endpoint response that is not XML.
            if response_text.lstrip("\ufeff \t\r\n").startswith("<"):
                records = _parse_pubmed_xml(response_text)
            else:
                records = _parse_medline(response_text)
                for record in records:
                    dep = record.get("DEP", "")
                    if isinstance(dep, str) and re.fullmatch(r"\d{8}", dep):
                        record["publication_date"] = f"{dep[:4]}-{dep[4:6]}-{dep[6:]}"
                    elif not dep:
                        dp = record.get("DP", "")
                        if isinstance(dp, str):
                            match = re.fullmatch(r"(\d{4})(?: ([A-Za-z]+)(?: (\d{1,2}))?)?", dp)
                            if match:
                                year, month, day = match.groups()
                                number = _normalise_pubmed_month(month) if month else None
                                if not month or number:
                                    record["publication_date"] = year + (f"-{number}" if number else "") + (f"-{int(day):02d}" if day else "")
            if not records:
                raise ValueError(
                    "PubMed EFetch returned no parseable records for the requested PMID batch"
                )
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
            # Do not put user-derived search phrases into routine logs.
            logger.info(f"PubMed [{label}] search")
            pmids = self._fetch_pmids(q)
            logger.info(f"PubMed [{label}] → {len(pmids)} result(s)")
            return pmids

        def _strip_tiab(q: str) -> str:
            return re.sub(r"\[tiab\]", "", q).strip()

        def _recent_results(ids: list[str]) -> list[dict[str, Any]]:
            records = self._finalise(ids)
            today = datetime.now(timezone.utc).date()
            start = today - timedelta(days=self._publication_days)
            return [record for record in records if _publication_in_window(record.get("publication_date"), start, today)]

        pmids: list[str] = []

        # Mode 5 can receive several corpus/theme queries from the executor.
        # Fetch a small quota for each theme and interleave the results so the
        # first theme cannot consume the complete candidate budget.
        if self._pubmed_queries_configured:
            if not self._pubmed_queries:
                logger.warning("PubMed: no valid corpus/theme queries were configured")
                return []

            per_query_limit = max(
                1,
                (self._max_results + len(self._pubmed_queries) - 1) // len(self._pubmed_queries),
            )
            query_results: list[list[str]] = []
            for index, theme_query in enumerate(self._pubmed_queries, start=1):
                query = f"({theme_query}) AND {date_filter}"
                result = self._fetch_pmids_with_limit(query, per_query_limit)
                logger.info(
                    f"PubMed [theme {index}/{len(self._pubmed_queries)}] → "
                    f"{len(result)} result(s)"
                )
                query_results.append(result)

            # Round-robin de-duplication retains coverage across themes while
            # keeping the configured total candidate limit.
            selected: list[str] = []
            seen_pmids: set[str] = set()
            for offset in range(max((len(result) for result in query_results), default=0)):
                for result in query_results:
                    if offset >= len(result):
                        continue
                    pmid = str(result[offset])
                    if pmid in seen_pmids:
                        continue
                    seen_pmids.add(pmid)
                    selected.append(pmid)
                    if len(selected) >= self._max_results:
                        break
                if len(selected) >= self._max_results:
                    break

            if not selected:
                logger.warning("PubMed: no results found for corpus/theme queries")
                return []
            return _recent_results(selected)

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

            logger.info("PubMed [date-only] search")
            pmids = self._fetch_pmids(date_filter)
            logger.info(f"PubMed [date-only] → {len(pmids)} result(s)")
            if not pmids:
                logger.warning("PubMed: no results found for the date-only query")
                return []
            return _recent_results(pmids)

        # ── Level 1: strict query (1-2 AND concept groups, [tiab]) ──────────
        if self._pubmed_query:
            pmids = _try("strict", self._pubmed_query)
            if len(pmids) >= min_n:
                return _recent_results(pmids)

        # ── Level 2: strict without [tiab] → full-field search ──────────────
        if self._pubmed_query and "[tiab]" in self._pubmed_query:
            pmids_fb = _try("strict|no-tiab", _strip_tiab(self._pubmed_query))
            if len(pmids_fb) >= min_n:
                return _recent_results(pmids_fb)
            if len(pmids_fb) > len(pmids):
                pmids = pmids_fb   # keep the better count for later

        # ── Level 3: broad query (single OR group, [tiab]) ──────────────────
        if self._pubmed_broad_query:
            pmids_fb = _try("broad", self._pubmed_broad_query)
            if len(pmids_fb) >= min_n:
                return _recent_results(pmids_fb)
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
        return _recent_results(pmids)

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
            publication_date=raw_paper.get("publication_date"),
            publication_history=raw_paper.get("publication_history", []),
        )
