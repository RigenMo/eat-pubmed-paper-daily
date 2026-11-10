from datetime import timedelta
from html.parser import HTMLParser
import re

import requests

_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class _AbstractText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "br", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def fetch_public_records(pmids, max_requests=4):
    """Fetch public MED PMIDs only, with a bounded number of core requests."""
    ids = list(dict.fromkeys(str(p) for p in pmids if str(p).isdigit()))[:400]
    records = []
    for offset in range(0, min(len(ids), max(0, max_requests) * 100), 100):
        batch = ids[offset:offset + 100]
        query = "SRC:MED AND (" + " OR ".join("EXT_ID:" + pmid for pmid in batch) + ")"
        try:
            response = requests.get(_URL, params={"query": query, "format": "json", "resultType": "core", "pageSize": len(batch)}, timeout=20, allow_redirects=False)
            response.raise_for_status()
            if getattr(response, "is_redirect", False):
                raise ValueError("Unexpected public index redirect")
            results = response.json()["resultList"]["result"]
            if not isinstance(results, list):
                raise ValueError("Invalid public index results")
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
        for item in results:
            if not isinstance(item, dict) or item.get("source") != "MED":
                continue
            pmid = str(item.get("id", ""))
            if pmid not in batch:
                continue
            journal_info = item.get("journalInfo") or {}
            journal_data = journal_info.get("journal") or {} if isinstance(journal_info, dict) else {}
            journal = journal_data.get("title", "") if isinstance(journal_data, dict) else ""
            first = item.get("firstPublicationDate", "")
            # Europe PMC can infer the day of a partial date. Require an
            # independently stated full electronic or print date to agree.
            explicit = (item.get("electronicPublicationDate"), item.get("printPublicationDate"))
            if not isinstance(first, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", first) or first not in explicit:
                continue
            parser = _AbstractText()
            parser.feed(str(item.get("abstractText") or ""))
            abstract = "\n\n".join(" ".join(part.split()) for part in "".join(parser.parts).splitlines() if part.strip())
            title = item.get("title") or ""
            if not isinstance(title, str) or not title.strip() or not abstract or not isinstance(journal, str) or not journal.strip():
                continue
            authors_data = (item.get("authorList") or {}).get("author", []) if isinstance(item.get("authorList") or {}, dict) else []
            authors = [a.get("fullName", "") for a in authors_data if isinstance(a, dict) and a.get("fullName")]
            doi = item.get("doi") or ""
            records.append({"PMID": pmid, "TI": title, "AB": abstract, "JT": journal, "AU": authors,
                            "LID": f"{doi} [doi]" if doi else "", "publication_date": first,
                            "publication_history": [{"event": "publication", "date": first, "source": "https://europepmc.org/article/MED/" + pmid}],
                            "metadata_source": "europe_pmc_core"})
    return records


def search_public_index(today, days, limit, focus_mode):
    """Search only fixed public topics and dates; never accept private input."""
    start = today - timedelta(days=days)
    base = f"SRC:MED AND FIRST_PDATE:[{start.isoformat()} TO {today.isoformat()}]"
    searches = []
    if focus_mode == "ukb_multimorbidity" and limit >= 3:
        searches.append((base + ' AND (TITLE:"UK Biobank" OR ABSTRACT:"UK Biobank" OR TITLE:UKB OR ABSTRACT:UKB OR TITLE:multimorbidity OR ABSTRACT:multimorbidity)', limit // 3))
    searches.append((base, limit))
    batches = []
    audit = []
    last_error = None
    for index, (query, requested) in enumerate(searches, 1):
        size = min(requested, 800)
        try:
            response = requests.get(_URL, params={"query": query + " sort_date:y", "format": "json", "resultType": "idlist", "pageSize": size}, timeout=20, allow_redirects=False)
            response.raise_for_status()
            if getattr(response, "is_redirect", False):
                raise ValueError()
            payload = response.json()
            count = payload["hitCount"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError()
            results = payload.get("resultList", {}).get("result", [])
            if not isinstance(results, list) or (count > 0 and not results):
                raise ValueError()
            ids = [str(item["id"]) for item in results if isinstance(item, dict) and item.get("source") == "MED" and str(item.get("id", "")).isdigit()]
            if count > 0 and not ids:
                raise ValueError()
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
            last_error = exc
            audit.append({"query_index": index, "requested": size, "returned": 0, "failed": True})
            continue
        batches.append(ids)
        audit.append({"query_index": index, "requested": size, "returned": len(ids)})
    if not batches and last_error is not None:
        raise RuntimeError("Invalid public index response") from None
    merged = []
    seen = set()
    for offset in range(max((len(batch) for batch in batches), default=0)):
        for batch in batches:
            if offset < len(batch) and batch[offset] not in seen:
                seen.add(batch[offset])
                merged.append(batch[offset])
                if len(merged) == limit:
                    return merged, audit
    return merged, audit
