from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import zotero_arxiv_daily.journal_metrics as journal_metrics
from zotero_arxiv_daily.journal_metrics import EasyScholarClient


def _response(payload, status_code=200):
    return SimpleNamespace(status_code=status_code, json=lambda: payload)


def _payload(all_rank=None, select_rank=None, code=200):
    return {
        "code": code,
        "data": {
            "officialRank": {
                "all": all_rank or {},
                "select": select_rank or {},
            }
        },
    }


def test_lookup_maps_official_all_rank_and_preserves_raw_categories(monkeypatch):
    calls = []
    payload = _payload(
        {
            "sciif": "25.0",
            "sci": "Q1",
            "sciUp": "医学1区",
            "sciUpSmall": "细胞生物学1区/老年医学1区/神经科学1区。",
            "sciif5": "26.0",
            "jci": "4.66",
        },
        {
            "sciif": "999.0",
            "sci": "Q4",
            "sciUp": "错误大类",
            "sciUpSmall": "错误小类",
        },
    )

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _response(payload)

    monkeypatch.setattr(journal_metrics.requests, "get", fake_get)
    client = EasyScholarClient("secret-value")

    result = client.lookup("  Nature   Aging ")

    assert result is not None
    assert result["jif"] == 25.0
    assert result["jif_year"] is None
    assert result["jcr_quartile"] == "Q1"
    assert result["jcr_year"] is None
    assert result["cas_quartile"] == {
        "big": "医学1区",
        "small": "细胞生物学1区/老年医学1区/神经科学1区。",
    }
    assert result["cas_year"] is None
    assert result["source"] == "https://www.easyscholar.cc/console/query"
    assert result["jcr_source"] == result["source"]
    assert result["cas_source"] == result["source"]
    assert result["metrics_provider"] == "EasyScholar"
    assert result["queried_at"] == datetime.now(timezone.utc).date().isoformat()

    assert len(calls) == 1
    assert calls[0][0] == "https://www.easyscholar.cc/open/getPublicationRank"
    assert calls[0][1] == {
        "params": {"secretKey": "secret-value", "publicationName": "Nature Aging"},
        "timeout": 10.0,
        "allow_redirects": False,
    }


@pytest.mark.parametrize("value", [None, "", "not-a-number", "NaN", "inf", "-inf", -1, float("nan"), float("inf")])
def test_invalid_current_jif_is_omitted_without_using_five_year_if(monkeypatch, value):
    calls = []
    payload = _payload({"sciif": value, "sciif5": "26.0", "sci": "Q1"})
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _response(payload),
    )

    result = EasyScholarClient("key").lookup("Nature Aging")

    assert result is not None
    assert "jif" not in result
    assert "jif_year" not in result
    assert len(calls) == 1


def test_absent_rank_fields_do_not_create_nonfinite_values(monkeypatch):
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: _response(_payload({"sciif": "not-a-number"})),
    )

    result = EasyScholarClient("key").lookup("Nature Aging")

    assert result is None


def test_empty_key_or_title_skips_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: calls.append(args))

    client = EasyScholarClient("")
    assert client.lookup("Nature Aging") is None
    assert EasyScholarClient("key").lookup("   ") is None
    assert calls == []


def test_same_normalized_title_is_queried_once_and_caches_success(monkeypatch):
    calls = []
    response = _response(_payload({"sciif": "25.0"}))
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)) or response,
    )
    client = EasyScholarClient("key")

    first = client.lookup("Nature Aging")
    second = client.lookup(" nature   aging ")

    assert first == second
    assert len(calls) == 1


def test_failure_is_cached_and_business_failure_disables_remaining_run(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _response({"code": 40002}, status_code=200)

    monkeypatch.setattr(journal_metrics.requests, "get", fake_get)
    client = EasyScholarClient("secret-value")

    assert client.lookup("Nature Aging") is None
    assert client.lookup(" nature aging ") is None
    assert client.lookup("Nature Food") is None
    assert len(calls) == 1


def test_an_unknown_business_error_does_not_block_the_next_journal(monkeypatch):
    responses = iter([_response({"code": 40088}), _response(_payload({"sciif": "25.0"}))])
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(EasyScholarClient, "_pace", lambda self: None)
    client = EasyScholarClient("key")

    assert client.lookup("Missing Journal") is None
    assert client.lookup("Nature Aging")["jif"] == 25.0


def test_network_failure_is_cached_without_leaking_secret_or_exception(monkeypatch):
    calls = []
    messages = []
    secret = "secret-value"
    exception = RuntimeError(f"{secret} https://www.easyscholar.cc/open/getPublicationRank")

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise exception

    monkeypatch.setattr(journal_metrics.requests, "get", fake_get)
    monkeypatch.setattr(journal_metrics.logger, "warning", lambda message, *args: messages.append((message, args)))
    client = EasyScholarClient(secret)

    assert client.lookup("Nature Aging") is None
    assert client.lookup("nature aging") is None
    assert len(calls) == 1
    rendered = repr(messages)
    assert secret not in rendered
    assert "easyscholar.cc" not in rendered


def test_requests_are_spaced_after_any_previous_request(monkeypatch):
    clock = iter([10.0, 10.1])
    sleeps = []
    monkeypatch.setattr(journal_metrics, "monotonic", lambda: next(clock))
    monkeypatch.setattr(journal_metrics, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: _response(_payload({"sciif": "25.0"})),
    )
    client = EasyScholarClient("key")

    client.lookup("Nature Aging")
    client.lookup("Nature Food")

    assert sleeps == [pytest.approx(0.45)]


def _paper(journal, metrics=None):
    return SimpleNamespace(journal=journal, journal_metrics=metrics)


def _journal_config(**easyscholar):
    return {
        "enabled": True,
        "min_jif": 10.0,
        "easyscholar": {"enabled": True, "api_key": "key", **easyscholar},
    }


def test_enrich_merges_live_metrics_without_dropping_static_fields(monkeypatch):
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: _response(_payload({"sciif": "25.0", "sci": "Q1"})),
    )
    paper = _paper("Nature Aging", {"year": 2025, "jif": 24.0, "source": "publisher", "jcr_category": "old category"})

    retained = journal_metrics.enrich_journal_metrics([paper], _journal_config())

    assert retained == [paper]
    assert paper.journal_metrics["jif"] == 25.0
    assert paper.journal_metrics["jif_year"] is None
    assert paper.journal_metrics["source"] == "https://www.easyscholar.cc/console/query"
    assert paper.journal_metrics["year"] == 2025
    assert paper.journal_metrics["jcr_quartile"] == "Q1"
    assert paper.journal_metrics["jcr_category"] is None


def test_enrich_duplicate_journal_is_queried_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _response(_payload({"sciif": "25.0"})),
    )
    papers = [_paper("Nature Aging"), _paper(" nature   aging ")]

    retained = journal_metrics.enrich_journal_metrics(papers, _journal_config())

    assert retained == papers
    assert len(calls) == 1


def test_enrich_strict_gate_excludes_live_jif_below_threshold(monkeypatch):
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: _response(_payload({"sciif": "9.9"})),
    )
    paper = _paper("Nature Aging")

    assert journal_metrics.enrich_journal_metrics([paper], _journal_config()) == []


def test_enrich_api_error_preserves_original_metadata(monkeypatch):
    original = {"jif": 25.0, "year": 2025, "source": "publisher"}
    paper = _paper("Nature Aging", original.copy())
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))

    retained = journal_metrics.enrich_journal_metrics([paper], _journal_config())

    assert retained == [paper]
    assert paper.journal_metrics == original


def test_enrich_missing_live_jif_does_not_clobber_fallback_source_or_year(monkeypatch):
    original = {"jif": 25.0, "year": 2025, "source": "publisher"}
    paper = _paper("Nature Aging", original.copy())
    monkeypatch.setattr(
        journal_metrics.requests,
        "get",
        lambda *args, **kwargs: _response(_payload({"sci": "Q1"})),
    )

    retained = journal_metrics.enrich_journal_metrics([paper], _journal_config())

    assert retained == [paper]
    assert paper.journal_metrics["jif"] == 25.0
    assert paper.journal_metrics["year"] == 2025
    assert paper.journal_metrics["source"] == "publisher"
    assert "jif_year" not in paper.journal_metrics


def test_enrich_disabled_or_keyless_returns_original_without_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: calls.append(args))
    papers = [_paper("Nature Aging", {"jif": 25.0})]

    assert journal_metrics.enrich_journal_metrics(papers, {"enabled": False}) is papers
    assert journal_metrics.enrich_journal_metrics(
        papers, {"enabled": True, "min_jif": 10, "easyscholar": {"enabled": True, "api_key": ""}}
    ) is papers
    assert calls == []


def test_enrich_can_add_metadata_when_the_strict_filter_is_disabled(monkeypatch):
    monkeypatch.setattr(
        journal_metrics.requests, "get",
        lambda *args, **kwargs: _response(_payload({"sciif": "9.9", "sci": "Q2"})),
    )
    config = _journal_config()
    config["enabled"] = False
    paper = _paper("Example Journal")

    assert journal_metrics.enrich_journal_metrics([paper], config) == [paper]
    assert paper.journal_metrics["jif"] == 9.9


def test_api_response_reaches_the_email_badges_with_correct_provenance(monkeypatch):
    from tests.canned_responses import make_sample_paper
    from zotero_arxiv_daily.construct_email import render_email

    payload = _payload({
        "sciif": "25.0", "sci": "Q1", "sciUp": "医学1区",
        "sciUpSmall": "细胞生物学1区/老年医学1区/神经科学1区。",
    })
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: _response(payload))
    paper = make_sample_paper(journal="Nature Aging")
    paper.journal_metrics = {"jif": 24.0, "year": 2025, "source": "https://example.org"}

    html = render_email(journal_metrics.enrich_journal_metrics([paper], _journal_config()))

    assert "JIF(影响因子)：25.0" in html
    assert "JCR分区：Q1" in html
    assert "年份未提供" not in html
    assert "学科未提供" not in html
    assert "中科院分区大类（升级版）" in html
    assert "细胞生物学1区/老年医学1区/神经科学1区。" in html
    assert "25.0（2025）" not in html
    assert 'href="https://www.easyscholar.cc/console/query"' in html


def test_live_mode_does_not_silently_fall_back_to_the_old_journal_list(monkeypatch):
    config = _journal_config()
    config["mode"] = "easyscholar"
    paper = _paper("Nature Aging", {"jif": 25.0, "year": 2025})
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))

    with pytest.raises(RuntimeError, match="live journal filtering is incomplete"):
        journal_metrics.enrich_journal_metrics([paper], config)


def test_live_mode_requires_a_valid_current_jif_not_an_audited_old_value(monkeypatch):
    config = _journal_config()
    config["mode"] = "easyscholar"
    paper = _paper("Nature Aging", {"jif": 25.0, "year": 2025})
    monkeypatch.setattr(journal_metrics.requests, "get", lambda *args, **kwargs: _response(_payload({"sci": "Q1"})))

    assert journal_metrics.enrich_journal_metrics([paper], config) == []
