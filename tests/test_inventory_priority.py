import importlib.util
import io
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('inventory_priority', Path(__file__).parents[1] / 'scripts/build_reading_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup(monkeypatch):
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'test')


def response(runs):
    return io.BytesIO(json.dumps({'total_count': len(runs), 'workflow_runs': runs}).encode())


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
    assert calls == [5, 5, 5]
    assert check() is False and len(calls) == 3


def test_actual_sender_pauses_inventory(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: response([{'path': '.github/workflows/main.yml'}]))
    check = module.sender_priority_check()
    assert check() is True and check.reason == 'sender_active'


def test_unknown_state_is_bounded_and_fail_closed(monkeypatch):
    setup(monkeypatch)
    calls = []
    def fail(*a, **k):
        calls.append(True)
        raise TimeoutError()
    monkeypatch.setattr(module, 'urlopen', fail)
    check = module.sender_priority_check()
    assert check() is True and check.reason == 'sender_status_unavailable'
    assert len(calls) == 2
