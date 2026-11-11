import asyncio
import time
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


def test_only_healthy_model_receives_its_full_share():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return "ok"
    client = client_with(create)
    client.failed_until = {model: 100 for model in MODELS[:3]}
    client.preferred = MODELS[3]
    assert client.create(model=MODELS[0], timeout=120) == "ok"
    assert calls[0]["model"] == MODELS[3]
    assert calls[0]["timeout"] == 90


def test_real_client_async_deadline_cancels_and_closes(monkeypatch):
    calls = []
    closed = []
    cancelled = []
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            closed.append(True)
        async def create(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                try:
                    while True:
                        await asyncio.sleep(0.005)
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise
            return "ok"
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    real = openai.OpenAI(api_key="secret-test-key", base_url="https://example.test/v1", max_retries=0)
    client = FailoverClient(real, MODELS, total_timeout=0.4, model_timeout=0.1)
    start = time.monotonic()
    try:
        assert client.create(model=MODELS[0]) == "ok"
    finally:
        real.close()
    assert time.monotonic() - start < 2
    assert calls == MODELS[:2]
    assert cancelled == [True]
    assert len(closed) == 2


def test_preferred_healthy_model_keeps_full_timeout_when_others_recover():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return "ok"
    client = client_with(create, lambda: 1000)
    client.failed_until = {MODELS[0]: 900, MODELS[1]: 999}
    client.preferred = MODELS[3]
    assert client.create(model=MODELS[0], timeout=120) == "ok"
    assert calls[0]["model"] == MODELS[3]
    assert calls[0]["timeout"] == 90


def test_whole_generation_deadline_caps_request_and_blocks_new_call():
    calls = []
    now = [100.0]
    def create(**kwargs):
        calls.append(kwargs)
        return 'ok'
    client = client_with(create, lambda: now[0])
    client.deadline = 108.0
    assert client.create(model=MODELS[0], timeout=120) == 'ok'
    assert calls[0]['timeout'] <= 8
    now[0] = 109.0
    with pytest.raises(TimeoutError):
        client.create(model=MODELS[0])
    assert len(calls) == 1


def test_endpoint_outage_is_not_reprobed_until_cooldown():
    now, calls = [0.0], []
    def fail(**kwargs):
        calls.append(kwargs['model'])
        raise error(503)
    client = client_with(fail, lambda: now[0])
    for _ in range(2):
        with pytest.raises(openai.APIStatusError):
            client.create(model=MODELS[0])
    assert len(calls) == 4
    now[0] = 61
    with pytest.raises(openai.APIStatusError):
        client.create(model=MODELS[0])
    assert len(calls) == 8


def test_independent_backup_uses_own_model_and_remaining_budget():
    now, calls = [0.0], []
    def fail(**kwargs):
        now[0] += kwargs['timeout']
        raise error(503)
    def succeed(**kwargs):
        calls.append(kwargs)
        return 'backup-ok'
    client = client_with(fail, lambda: now[0])
    client.backup_client = FailoverClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=succeed))),
                                        ['backup-model'], clock=lambda: now[0])
    assert client.create(model=MODELS[0], timeout=120) == 'backup-ok'
    assert calls[0]['model'] == 'backup-model'
    assert calls[0]['timeout'] <= 60


def test_request_budget_expiry_does_not_mark_untried_endpoint_down():
    now, calls = [0.0], []
    def fail(**kwargs):
        calls.append(kwargs['model'])
        now[0] += 250
        raise error(503)
    client = client_with(fail, lambda: now[0])
    with pytest.raises(openai.APIStatusError):
        client.create(model=MODELS[0], timeout=120)
    assert calls == [MODELS[0]]
    assert client.endpoint_failed_until == 0


def test_optional_background_priority_never_calls_model():
    from zotero_arxiv_daily.llm_client import BackgroundWorkPaused
    client = client_with(lambda **kwargs: pytest.fail('sender priority must prevent model call'))
    client.priority_check = lambda: True
    with pytest.raises(BackgroundWorkPaused):
        client.create(model=MODELS[0])
    assert client.endpoint_failed_until == 0
