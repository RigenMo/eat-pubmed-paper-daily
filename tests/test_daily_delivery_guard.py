import io
import json
import zipfile
from datetime import datetime, timezone
from urllib.request import Request
import pytest
from scripts import daily_delivery_guard as guard

NOW = datetime(2026, 11, 3, 1, tzinfo=timezone.utc)


def receipt():
    return {'schema_version': 1, 'run_id': '5', 'run_attempt': '1', 'date': '2026-11-03', 'recorded_at_utc': '2026-11-02T17:00:00Z', 'mail_sent': True, 'paper_count': 1, 'complete_cards': 1, 'expected_cards': 1, 'quality_passed': True, 'papers': [{'url': 'x', 'journal': 'x', 'publication_date': 'x', 'jif': 1}]}


def setup_archive(monkeypatch, payload):
    monkeypatch.setattr(guard, 'api_pages', lambda *args: iter([{'name': 'daily-delivery-receipt-1', 'expired': False, 'created_at': '2026-11-02', 'id': 10}]))
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(payload)
    monkeypatch.setattr(guard, 'build_opener', lambda *args: Opener())


def archive(value):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr('delivery-receipt.json', json.dumps(value))
    return data.getvalue()


def test_receipt_validation(monkeypatch):
    setup_archive(monkeypatch, archive(receipt()))
    assert guard.load_delivery_receipt('owner/repo', 'secret', 5) == receipt()


@pytest.mark.parametrize('change', [{'run_id': '6'}, {'recorded_at_utc': '2026-11-02T17:00:00'}, {'quality_passed': False}])
def test_invalid_receipt(monkeypatch, change):
    value = receipt()
    value.update(change)
    setup_archive(monkeypatch, archive(value))
    with pytest.raises(ValueError):
        guard.load_delivery_receipt('owner/repo', 'secret', 5)


def test_bad_zip(monkeypatch):
    setup_archive(monkeypatch, b'bad')
    with pytest.raises(zipfile.BadZipFile):
        guard.load_delivery_receipt('owner/repo', 'secret', 5)


def test_redirect_strips_credentials():
    request = Request('https://api.github.com/a', headers={'Authorization': 'Bearer secret'})
    redirected = guard.SafeRedirect().redirect_request(request, None, 302, 'Found', {}, 'https://storage.example/a')
    assert not redirected.has_header('Authorization')


def test_status_receipts_same_run_and_supervisor(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}] if 'main.yml' in url else [{'id': 7}])
        if '/runs/5/' in url:
            return iter([{'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'cancelled'}]}])
        return iter([{'name': 'notify-on-failure', 'steps': [{'name': 'Send supervisor failure notice', 'conclusion': 'success', 'completed_at': '2026-11-02T17:00:00Z'}]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: receipt())
    assert guard.delivery_status('owner/repo', 'token', 5, NOW) == (True, True)
    old = receipt()
    old['date'] = '2026-11-02'
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: old)
    assert guard.delivery_status('owner/repo', 'token', 5, NOW) == (False, True)


def test_current_run_previous_success(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([])
        return iter([{'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T17:00:00Z'}, {'name': 'Run script', 'conclusion': None}]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', 5, NOW) == (True, False)


def test_api_error_propagates(monkeypatch):
    def fail(*args):
        raise OSError('API unavailable')
    monkeypatch.setattr(guard, 'api_pages', fail)
    with pytest.raises(OSError):
        guard.delivery_status('owner/repo', 'token', 5, NOW)


def test_previous_day_run_completed_today_and_previous_attempt(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            assert '2026-11-01T16%3A00%3A00Z' in url
            return iter([{'id': 1}, {'id': 3}] if 'main.yml' in url else [])
        if '/runs/1/' not in url:
            return iter([])
        assert '/runs/1/jobs?filter=all' in url
        return iter([
            {'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T16:01:00Z'}]},
            {'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'skipped'}]},
        ])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', 3, NOW) == (True, False)


def test_before_midnight_and_successful_notice(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}] if 'main.yml' in url else [])
        if '/runs/5/' not in url:
            return iter([])
        return iter([
            {'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T15:59:59Z'}]},
            {'name': 'notify-on-failure', 'steps': [{'name': 'Send failure notice', 'conclusion': 'success', 'completed_at': '2026-11-02T23:20:00Z'}]},
        ])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', 9, NOW) == (False, True)


def test_missing_completion_time_fails(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}] if 'main.yml' in url else [])
        return iter([{'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': None}]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    with pytest.raises((AttributeError, ValueError)):
        guard.delivery_status('owner/repo', 'token', 9, NOW)


def test_api_failure_propagates(monkeypatch):
    def fail(*args):
        raise OSError('API unavailable')
    monkeypatch.setattr(guard, 'api_pages', fail)
    with pytest.raises(OSError):
        guard.delivery_status('owner/repo', 'token', 9, NOW)


def test_watchdog_current_run_previous_notice(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([])
        assert '/runs/9/jobs?filter=all' in url
        return iter([{'name': 'notify-on-failure', 'steps': [
            {'name': 'Send supervisor failure notice', 'conclusion': 'success', 'completed_at': '2026-11-02T17:00:00Z'},
            {'name': 'Send supervisor failure notice', 'conclusion': None},
        ]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', 9, NOW) == (False, True)


def test_morning_notice_is_independent_and_not_delivery(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 7}] if 'keep-alive.yml' in url else [])
        return iter([
            {'name': 'notify-on-failure', 'steps': [{'name': 'Send supervisor failure notice', 'conclusion': 'success', 'completed_at': '2026-11-02T17:00:00Z'}]},
            {'name': 'morning-deadline', 'steps': [{'name': 'Send morning deadline notice', 'conclusion': 'success', 'completed_at': '2026-11-03T02:10:00Z'}]},
        ])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', None, NOW) == (False, True)
    assert guard.delivery_status('owner/repo', 'token', None, NOW, 'morning') == (False, True)
    assert guard.delivery_status('owner/repo', 'token', None, datetime(2026, 11, 2, 15, tzinfo=timezone.utc), 'morning') == (False, False)


def test_failure_notice_does_not_suppress_morning(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 7}] if 'keep-alive.yml' in url else [])
        return iter([{'name': 'notify-on-failure', 'steps': [{'name': 'Send supervisor failure notice', 'conclusion': 'success', 'completed_at': '2026-11-02T17:00:00Z'}]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', None, NOW, 'morning') == (False, False)


def test_in_progress_workflow_notice_and_queued_job(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            assert 'status=completed' not in url
            return iter([{'id': 7, 'status': 'in_progress'}] if 'keep-alive.yml' in url else [])
        return iter([
            {'name': 'queued', 'steps': None},
            {'name': 'morning-deadline', 'steps': [{'name': 'Send morning deadline notice', 'conclusion': 'success', 'completed_at': '2026-11-03T02:10:00Z'}]},
        ])
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    assert guard.delivery_status('owner/repo', 'token', None, NOW, 'morning') == (False, True)


def test_progress_combines_receipts_and_deduplicates(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            if 'main.yml' in url:
                return iter([{'id': 5}])
            if 'supplement.yml' in url:
                return iter([{'id': 6}])
            return iter([])
        return iter([])
    def get_receipt(repo, token, run_id):
        value = receipt()
        ids = range(8) if run_id == '5' else (7, 8, 9)
        value.update(paper_count=len(ids), complete_cards=len(ids), papers=[{'url': str(i)} for i in ids])
        return value
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', get_receipt)
    progress = {}
    assert guard.delivery_status('owner/repo', 'token', None, NOW, progress=progress) == (True, False)
    assert progress == {'paper_count': 10, 'day_complete': True, 'delivery_unknown': False}


def test_incomplete_cards_and_missing_receipt_are_unknown(monkeypatch):
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}] if 'main.yml' in url else [])
        return iter([{'name': 'calculate-and-send', 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T17:00:00Z'}]}])
    monkeypatch.setattr(guard, 'api_pages', pages)
    value = receipt()
    value['complete_cards'] = 0
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: value)
    progress = {}
    assert guard.delivery_status('owner/repo', 'token', None, NOW, progress=progress)[0]
    assert progress == {'paper_count': 0, 'day_complete': False, 'delivery_unknown': True}
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: None)
    guard.delivery_status('owner/repo', 'token', None, NOW, progress=progress)
    assert progress['delivery_unknown'] and not progress['day_complete']


def test_previous_day_receipt_precedes_today_step_but_attempt_mismatch_unknown(monkeypatch):
    job = {'name': 'calculate-and-send', 'run_attempt': 1, 'steps': [{'name': 'Run script', 'conclusion': 'success', 'completed_at': '2026-11-02T16:01:00Z'}]}
    def pages(url, token, key):
        if key == 'workflow_runs':
            return iter([{'id': 5}] if 'main.yml' in url else [])
        return iter([job])
    old = receipt()
    old['date'] = '2026-11-02'
    monkeypatch.setattr(guard, 'api_pages', pages)
    monkeypatch.setattr(guard, 'load_delivery_receipt', lambda *args: old)
    progress = {}
    assert guard.delivery_status('owner/repo', 'token', None, NOW, progress=progress) == (False, False)
    assert not progress['delivery_unknown'] and progress['paper_count'] == 0
    job['run_attempt'] = 2
    assert guard.delivery_status('owner/repo', 'token', None, NOW, progress=progress)[0]
    assert progress['delivery_unknown']
