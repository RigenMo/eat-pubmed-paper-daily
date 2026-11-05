"""Temporary topic emphasis and the user's cognitive-innovation preference."""
import re
from datetime import date, datetime, timedelta, timezone

_SHANGHAI = timezone(timedelta(hours=8))
_FOCUS = re.compile(
    r"\bUK\s+Biobank\b|\bUKB\b|\bmultimorbid\w*\b|"
    r"\bmultiple (?:chronic|long[- ]term) conditions\b", re.IGNORECASE
)
_INNOVATION_ORDER = {"cognitive": 0, "mixed": 1, "capability": 2, "uncertain": 3}


def effective_focus_mode(settings, today=None):
    mode = str(settings.get("focus_mode", "balanced") or "balanced")
    if mode not in {"balanced", "ukb_multimorbidity"}:
        raise ValueError("Recommendation focus must be balanced or ukb_multimorbidity")
    until = settings.get("focus_until")
    if until:
        today = today or datetime.now(_SHANGHAI).date()
        if today > date.fromisoformat(str(until)):
            return "balanced"
    return mode


def focus_queries(mode):
    if mode != "ukb_multimorbidity":
        return []
    return [
        '(("UK Biobank"[tiab] OR UKB[tiab]) AND (multimorbidity[tiab] OR "multiple chronic conditions"[tiab] OR "disease trajectories"[tiab]))',
        '((multimorbidity[tiab] OR "multiple long-term conditions"[tiab]) AND (cohort[tiab] OR longitudinal[tiab]))',
        '(("UK Biobank"[tiab] OR UKB[tiab]) AND (aging[tiab] OR healthspan[tiab] OR "disease progression"[tiab]))',
    ]


def focus_evidence(paper, mode):
    if mode != "ukb_multimorbidity":
        return None
    for material in (paper.title or "", paper.abstract or ""):
        match = _FOCUS.search(material)
        if match:
            # An explicit user interest is a separate basis, never a fake Zotero citation.
            quote = material[max(0, match.start() - 35):match.end() + 70]
            return {
                "basis": "current_focus", "reason": "涉及 UK Biobank 或多病共存，与当前课题主题相符。",
                "candidate_quote": quote,
            }
    return None


def order_candidates(papers, mode):
    ordered = sorted(papers, key=lambda paper: paper.publication_date or "", reverse=True)
    ordered.sort(key=lambda paper: (
        _INNOVATION_ORDER.get((paper.innovation_assessment or {}).get("type"), 3),
        0 if focus_evidence(paper, mode) else 1,
    ))
    return ordered
