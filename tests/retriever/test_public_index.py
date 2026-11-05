from datetime import date
from types import SimpleNamespace
import pytest
from zotero_arxiv_daily.retriever import public_index


def test_public_search_uses_fixed_queries_and_interleaves(monkeypatch):
    calls = []
    def get(url, params, timeout):
        calls.append(params)
        records = ([{"id": "1", "source": "MED"}, {"id": "bad", "source": "MED"}] if len(calls) == 1 else [{"id": "2", "source": "MED"}, {"id": "1", "source": "MED"}, {"id": "3", "source": "PMC"}])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"hitCount": 2, "resultList": {"result": records}})
    monkeypatch.setattr(public_index.requests, "get", get)
    ids, audit = public_index.search_public_index(date(2026, 11, 5), 90, 6, "ukb_multimorbidity")
    assert ids == ["1", "2"]
    assert len(audit) == 2
    assert all('PRIVATE' not in c['query'] and '2026-08-07 TO 2026-11-05' in c['query'] and 'sort_date:y' in c['query'] for c in calls)
    assert [c['pageSize'] for c in calls] == [2, 6]


def test_balanced_search_and_invalid_response(monkeypatch):
    calls = []
    def get(url, params, timeout):
        calls.append(params)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    monkeypatch.setattr(public_index.requests, "get", get)
    with pytest.raises(RuntimeError, match="Invalid public index response"):
        public_index.search_public_index(date(2026, 11, 5), 90, 900, "balanced")
    assert len(calls) == 1 and calls[0]['pageSize'] == 800
    assert 'UKB' not in calls[0]['query']


def test_nonempty_result_without_valid_medline_ids_is_not_a_genuine_zero(monkeypatch):
    monkeypatch.setattr(public_index.requests, 'get', lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda:None,
        json=lambda:{'hitCount':1,'resultList':{'result':[{'source':'PMC','id':'PMC1'}]}}
    ))
    with pytest.raises(RuntimeError, match='Invalid public index response'):
        public_index.search_public_index(date(2026,11,5),90,10,'balanced')
