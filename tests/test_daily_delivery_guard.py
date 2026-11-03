from datetime import datetime, timezone
import pytest
from scripts import daily_delivery_guard as guard

NOW = datetime(2026, 11, 3, 1, tzinfo=timezone.utc)


def test_previous_day_run_completed_today_and_previous_attempt(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            assert '2026-11-01T16%3A00%3A00Z' in url
            return iter([{'id': 1}, {'id': 3}])
        assert '/runs/1/jobs?filter=all' in url
        return iter([
            {'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T16:01:00Z'}]},
            {'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'skipped'}]},
        ])
    monkeypatch.setattr(guard, 'api_pages', pages)
    assert guard.delivery_status('owner/repo', 'token', 3, NOW) == (True, False)


def test_before_midnight_and_successful_notice(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}])
        return iter([
            {'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T15:59:59Z'}]},
            {'name': 'notify-on-failure', 'steps': [{'name': 'Send failure notice', 'conclusion': 'success', 'completed_at': '2026-11-02T23:20:00Z'}]},
        ])
    monkeypatch.setattr(guard, 'api_pages', pages)
    assert guard.delivery_status('owner/repo', 'token', 9, NOW) == (False, True)


def test_missing_completion_time_fails(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}])
        return iter([{'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': None}]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    with pytest.raises((AttributeError, ValueError)):
        guard.delivery_status('owner/repo', 'token', 9, NOW)


def test_api_failure_propagates(monkeypatch):
    def fail(*args):
        raise OSError('API unavailable')
    monkeypatch.setattr(guard, 'api_pages', fail)
    with pytest.raises(OSError):
        guard.delivery_status('owner/repo', 'token', 9, NOW)

