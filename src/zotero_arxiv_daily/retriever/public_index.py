from datetime import timedelta

import requests

_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


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
    for index, (query, requested) in enumerate(searches, 1):
        size = min(requested, 800)
        response = requests.get(_URL, params={"query": query + " sort_date:y", "format": "json", "resultType": "idlist", "pageSize": size}, timeout=30)
        response.raise_for_status()
        try:
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
        except (ValueError, TypeError, KeyError, AttributeError):
            raise RuntimeError("Invalid public index response") from None
        batches.append(ids)
        audit.append({"query_index": index, "requested": size, "returned": len(ids)})
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
