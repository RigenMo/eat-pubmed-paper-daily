import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

spec = importlib.util.spec_from_file_location('inventory_priority', Path(__file__).parents[1] / 'scripts/build_reading_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup(monkeypatch):
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'test')


def response(runs, total=None):
    return io.BytesIO(json.dumps({'total_count': len(runs) if total is None else total, 'workflow_runs': runs}).encode())


def test_transient_read_error_retries_before_pausing(monkeypatch):
    setup(monkeypatch)
    calls = []
    def open_request(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError()
        return response([])
    monkeypatch.setattr(module, 'urlopen', open_request)
    check = module.sender_priority_check()
    assert check() is False and check.reason == 'sender_idle'
    assert calls == [5] * 7
    assert check() is False and len(calls) == 7


def test_actual_sender_pauses_inventory(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: response([{'path': '.github/workflows/main.yml@main', 'status': 'in_progress'}], total=257))
    check = module.sender_priority_check()
    assert check() is True and check.reason == 'sender_active'


def test_unknown_state_is_bounded_and_fail_closed(monkeypatch, capsys):
    setup(monkeypatch)
    calls = []
    def fail(*a, **k):
        calls.append(True)
        raise TimeoutError('https://secret.invalid/?token=test')
    monkeypatch.setattr(module, 'urlopen', fail)
    check = module.sender_priority_check()
    assert check() is True and check.reason == 'sender_status_unavailable'
    assert len(calls) == 2
    diagnostic = capsys.readouterr().out
    assert 'endpoint=sender_status' in diagnostic and 'stage=request' in diagnostic
    assert 'network_or_timeout' in diagnostic and 'secret.invalid' not in diagnostic and 'token=' not in diagnostic


def test_queries_exclude_unrelated_workflow_pagination(monkeypatch):
    setup(monkeypatch)
    calls = []
    def open_request(request, timeout):
        calls.append(request.full_url)
        assert request.get_method() == 'GET'
        assert timeout == 5
        return response([])
    monkeypatch.setattr(module, 'urlopen', open_request)
    check = module.sender_priority_check()
    assert check() is False and check.reason == 'sender_idle'
    assert calls == [f'https://api.github.com/repos/owner/repo/actions/workflows/{workflow}/runs?status={status}&per_page=1'
                     for workflow in ('main.yml', 'supplement.yml', 'test.yml') for status in ('in_progress', 'queued')]


@pytest.mark.parametrize('runs,total', [([], 1), ([{'path': '.github/workflows/main.yml', 'status': 'in_progress'}], 0),
                                      ([{'path': '.github/workflows/unrelated.yml', 'status': 'in_progress'}], 1),
                                      ([{'path': '.github/workflows/main.yml', 'status': 'completed'}], 1), ([None], 1), ([], -1)])
def test_contradictory_state_fails_closed_without_retry(monkeypatch, capsys, runs, total):
    setup(monkeypatch)
    calls = []
    def open_request(*args, **kwargs):
        calls.append(True)
        return response(runs, total)
    monkeypatch.setattr(module, 'urlopen', open_request)
    check = module.sender_priority_check()
    assert check() is True and check.reason == 'sender_status_unavailable'
    assert len(calls) == 1
    assert 'stage=validate' in capsys.readouterr().out


@pytest.mark.parametrize('code,expected_calls', [(429, 7), (503, 7), (403, 1)])
def test_only_transient_http_reads_retry(monkeypatch, code, expected_calls):
    setup(monkeypatch)
    calls = []
    def open_request(request, timeout):
        calls.append(request.get_method())
        if len(calls) == 1:
            raise HTTPError(request.full_url, code, 'secret response text', {}, None)
        return response([])
    monkeypatch.setattr(module, 'urlopen', open_request)
    check = module.sender_priority_check()
    assert check() is (code == 403)
    assert len(calls) == expected_calls and set(calls) == {'GET'}
