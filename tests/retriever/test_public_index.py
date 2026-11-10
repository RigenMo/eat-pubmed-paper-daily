from datetime import date
from types import SimpleNamespace
import pytest
from zotero_arxiv_daily.retriever import public_index


def test_public_search_uses_fixed_queries_and_interleaves(monkeypatch):
    calls = []
    def get(url, params, timeout, **kwargs):
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
    def get(url, params, timeout, **kwargs):
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


def test_core_details_are_bounded_public_and_keep_complete_metadata(monkeypatch):
    calls = []
    def get(url, params, timeout, allow_redirects):
        calls.append((url, params, timeout, allow_redirects))
        item = {"id": "42", "source": "MED", "title": "A study", "doi": "10.1234/example",
                "abstractText": "<p><b>Background:</b> First.</p><p>Second.</p>",
                "journalInfo": {"journal": {"title": "A journal"}},
                "authorList": {"author": [{"fullName": "Smith J"}]},
                "firstPublicationDate": "2026-11-01", "electronicPublicationDate": "2026-11-01"}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"resultList": {"result": [item]}})
    monkeypatch.setattr(public_index.requests, "get", get)
    records = public_index.fetch_public_records(["42", "PRIVATE", *map(str, range(1000))], max_requests=1)
    assert len(calls) == 1 and calls[0][2:] == (20, False)
    assert "PRIVATE" not in str(calls) and calls[0][1]["pageSize"] == 100
    assert records[0]["AB"] == "Background: First.\n\nSecond."
    assert records[0]["LID"] == "10.1234/example [doi]"
    assert records[0]["JT"] == "A journal" and records[0]["AU"] == ["Smith J"]


def test_core_rejects_unconfirmed_date_and_missing_abstract(monkeypatch):
    base = {"id": "42", "source": "MED", "title": "Title", "abstractText": "Abstract",
            "journalInfo": {"journal": {"title": "Journal"}, "printPublicationDate": "2026-11-01"},
            "firstPublicationDate": "2026-11-01"}
    monkeypatch.setattr(public_index.requests, "get", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"resultList": {"result": [base]}}))
    assert public_index.fetch_public_records(["42"]) == []
    base["electronicPublicationDate"] = "2026-11-01"
    base["abstractText"] = ""
    assert public_index.fetch_public_records(["42"]) == []


def test_public_search_keeps_first_channel_when_second_fails(monkeypatch):
    calls = []
    def get(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise public_index.requests.Timeout()
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"hitCount": 1, "resultList": {"result": [{"id": "1", "source": "MED"}]}})
    monkeypatch.setattr(public_index.requests, "get", get)
    ids, audit = public_index.search_public_index(date(2026, 11, 5), 90, 6, "ukb_multimorbidity")
    assert ids == ["1"] and audit[1]["failed"] is True
