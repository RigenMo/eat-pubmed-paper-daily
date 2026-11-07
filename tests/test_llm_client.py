from types import SimpleNamespace

import httpx
import openai
import pytest
from loguru import logger

from zotero_arxiv_daily.llm_client import FailoverClient

MODELS = ["kimi-k3", "glm-5.3", "deepseek-v4.1-flash", "qwen3.8-max"]


def error(status, code="error"):
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status, request=request)
    return openai.APIStatusError("secret prompt and key", response=response, body={"error": {"code": code}})


def client_with(action, clock=lambda: 0):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=action)))
    return FailoverClient(client, MODELS, clock=clock)


def test_failover_sticky_cooldown_and_budget():
    now = [0]
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == MODELS[0]:
            now[0] += kwargs["timeout"]
            raise error(503)
        return "ok"
    client = client_with(create, lambda: now[0])
    assert client.chat.completions.create(model=MODELS[0], timeout=120) == "ok"
    assert [call["model"] for call in calls] == MODELS[:2]
    assert calls[0]["timeout"] == 30
    assert calls[1]["timeout"] == 30
    client.chat.completions.create(model=MODELS[0])
    assert calls[-1]["model"] == MODELS[1]


def test_connection_timeout_changes_model():
    calls = []
    def create(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.test"))
        return "ok"
    assert client_with(create).create(model=MODELS[0]) == "ok"
    assert calls == MODELS[:2]


@pytest.mark.parametrize("status,code", [(401, "invalid_api_key"), (429, "insufficient_quota"), (429, "insufficient_user_quota")])
def test_credentials_and_quota_fail_fast(status, code):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise error(status, code)
    with pytest.raises(openai.APIStatusError):
        client_with(create).create(model=MODELS[0])
    assert len(calls) == 1


def test_all_fail_preserves_error_without_logging_secrets():
    logs = []
    sink = logger.add(logs.append)
    errors = []
    def create(**kwargs):
        exc = error(503)
        errors.append(exc)
        raise exc
    try:
        with pytest.raises(openai.APIStatusError) as caught:
            client_with(create).create(model=MODELS[0])
    finally:
        logger.remove(sink)
    assert caught.value is errors[-1]
    assert len(errors) == 4
    assert "secret prompt and key" not in "".join(logs)


def test_unknown_errors_do_not_retry():
    def create(**kwargs):
        raise ValueError("bad code")
    with pytest.raises(ValueError):
        client_with(create).create(model=MODELS[0])


@pytest.mark.parametrize("status", [408, 500, 529])
def test_transient_status_changes_model(status):
    calls = []
    def create(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise error(status)
        return "ok"
    assert client_with(create).create(model=MODELS[0]) == "ok"
    assert calls == MODELS[:2]


def test_all_cooling_probes_later_recovered_model():
    calls = []
    def create(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] != MODELS[2]:
            raise error(503)
        return "ok"
    client = client_with(create)
    client.failed_until = {model: (i + 1) * 100 for i, model in enumerate(MODELS)}
    assert client.create(model=MODELS[0], timeout=120) == "ok"
    assert calls == MODELS[:3]


def test_healthy_failure_probes_cooling_fallback():
    calls = []
    def create(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] != MODELS[2]:
            raise error(503)
        return "ok"
    client = client_with(create)
    client.failed_until = {MODELS[0]: 100, MODELS[2]: 200, MODELS[3]: 300}
    client.preferred = MODELS[1]
    assert client.create(model=MODELS[0]) == "ok"
    assert calls == [MODELS[1], MODELS[0], MODELS[2]]


def test_explicit_primary_outside_fallback_list_is_preserved():
    calls = []
    def create(**kwargs):
        calls.append(kwargs["model"])
        return "ok"
    assert client_with(create).create(model="configured-primary") == "ok"
    assert calls == ["configured-primary"]
