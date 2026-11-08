import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError

import pytest
from scripts import sent_paper_history as history


def http_error(code):
    return HTTPError('url', code, 'error', {}, None)


def empty_history():
    return {'version': 1, 'entries': {}, 'reconciled_at': '2026-01-01T00:00:00+00:00', 'retention_days': 180}


def test_stdlib_import():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(['src', '.']))
    result = subprocess.run([sys.executable, '-S', '-c', 'import scripts.sent_paper_history'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_branch_bootstrap(monkeypatch):
    calls = []
    def request(method, path, payload=None):
        calls.append((method, path, payload))
        if path.startswith('contents/') and method == 'GET':
            raise http_error(404)
        if path == f'git/ref/heads/{history.BRANCH}':
            raise http_error(404)
        if path == '':
            return {'default_branch': 'main'}
        if path == 'git/ref/heads/main':
            return {'object': {'sha': 'abc'}}
        return {}
    monkeypatch.setattr(history, 'request', request)
    assert history.read_remote() == (None, None)
    assert all(call[0] == 'GET' for call in calls)
    history.persist({})
    assert any(call[0] == 'POST' and call[1] == 'git/refs' for call in calls)
    assert any(call[0] == 'PUT' for call in calls)


def test_existing_branch_missing_file_fails(monkeypatch):
    def request(method, path, payload=None):
        if path.startswith('contents/'):
            raise http_error(404)
        return {}
    monkeypatch.setattr(history, 'request', request)
    with pytest.raises(ValueError, match='missing'):
        history.read_remote()


def test_save_no_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(history, 'persist', lambda *a: pytest.fail('unexpected write'))
    history.save()


@pytest.mark.parametrize('change', [{'mail_sent': False}, {'run_id': 'other'}])
def test_invalid_receipt_cannot_write(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')
    path = tmp_path / 'outputs' / 'delivery-receipt.json'
    path.parent.mkdir()
    receipt = {'schema_version': 1, 'run_id': '1', 'run_attempt': '1', 'mail_sent': True,
               'recorded_at_utc': '2026-11-08T00:00:00+00:00', 'date': '2026-11-08',
               'paper_count': 1, 'complete_cards': 1, 'expected_cards': 1, 'quality_passed': True,
               'papers': [{'url': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'journal': 'J', 'publication_date': '2026', 'jif': 1}]}
    receipt.update(change)
    path.write_text(json.dumps(receipt), encoding='utf-8')
    monkeypatch.setattr(history, 'persist', lambda *a: pytest.fail('unexpected write'))
    with pytest.raises(ValueError):
        history.save()


def test_cas_preserves_concurrent_entry_and_expires(monkeypatch):
    today = datetime.now(timezone.utc).date()
    first = empty_history()
    second = empty_history()
    second['entries'] = {'a' * 64: today.isoformat(), 'b' * 64: (today - timedelta(days=181)).isoformat()}
    states = iter([(first, 'old'), (second, 'new')])
    monkeypatch.setattr(history, 'read_remote', lambda: next(states))
    writes = []
    def request(method, path, payload=None):
        writes.append(payload)
        if len(writes) == 1:
            raise http_error(409)
        return {}
    monkeypatch.setattr(history, 'request', request)
    result = history.persist({'c' * 64: today.isoformat()})
    assert writes[-1]['sha'] == 'new'
    assert set(result['entries']) == {'a' * 64, 'c' * 64}


def test_prepare_only_confirmed_receipts_and_latest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('GITHUB_REPOSITORY', 'a/b')
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_RUN_ID', '3')
    monkeypatch.setattr(history, 'read_remote', lambda: (None, None))
    monkeypatch.setattr(history, 'api_pages', lambda *a: [{'id': i, 'created_at': '2020-01-01T00:00:00Z'} for i in (1, 2, 3, 4)])
    def receipt(repo, token, run, all_attempts=False):
        if run == '4':
            return []
        return [{'date': '2026-11-08' if run == '1' else '2026-11-07', 'papers': [{'url': 'https://pubmed.ncbi.nlm.nih.gov/123/'}]}]
    monkeypatch.setattr(history, 'load_delivery_receipt', receipt)
    saved = []
    def persist(entries, stamp):
        saved.append(entries)
        return empty_history()
    monkeypatch.setattr(history, 'persist', persist)
    history.prepare(tmp_path / 'history.json')
    assert list(saved[0].values()) == ['2026-11-08']
    assert 'unavailable receipts: 3' in capsys.readouterr().out


def test_invalid_history_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(history, 'read_remote', lambda: (_ for _ in ()).throw(ValueError('bad history')))
    with pytest.raises(ValueError):
        history.prepare(tmp_path / 'history.json')
    assert not (tmp_path / 'history.json').exists()


def test_valid_local_receipt_writes_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')
    path = tmp_path / 'outputs' / 'delivery-receipt.json'
    path.parent.mkdir()
    receipt = {'schema_version': 1, 'run_id': '1', 'run_attempt': '1', 'mail_sent': True,
               'recorded_at_utc': '2026-11-08T00:00:00+00:00', 'date': '2026-11-08',
               'paper_count': 1, 'complete_cards': 1, 'expected_cards': 1, 'quality_passed': True,
               'papers': [{'url': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'journal': 'J', 'publication_date': '2026', 'jif': 1}]}
    path.write_text(json.dumps(receipt), encoding='utf-8')
    saved = []
    monkeypatch.setattr(history, 'persist', lambda entries: saved.append(entries))
    history.save()
    assert len(saved[0]) == 1
    assert all(len(key) == 64 and value == '2026-11-08' for key, value in saved[0].items())


@pytest.mark.parametrize('conclusion,log,blocked', [
    ('success', '', True), ('failure', 'Sending email...', True),
    ('failure', 'SUPPLEMENT_SMTP_STARTED', True), ('failure', 'failed before SMTP', False),
    ('skipped', '', False),
])
def test_missing_receipt_smtp_check(monkeypatch, conclusion, log, blocked):
    from io import BytesIO
    from types import SimpleNamespace
    job = {'name': 'calculate-and-send', 'run_attempt': 1, 'id': 42,
           'steps': [{'name': 'Run script', 'conclusion': conclusion}]}
    monkeypatch.setattr(history, 'api_pages', lambda *a: [job])
    monkeypatch.setattr(history, 'build_opener', lambda *a: SimpleNamespace(open=lambda *a, **kw: BytesIO(log.encode())))
    if blocked:
        with pytest.raises(RuntimeError):
            history.verify_missing_receipts('a/b', 'token', '1', 'main.yml', [])
    else:
        history.verify_missing_receipts('a/b', 'token', '1', 'main.yml', [])


def test_prepare_collects_multiple_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_REPOSITORY', 'a/b')
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setattr(history, 'read_remote', lambda: (None, None))
    monkeypatch.setattr(history, 'api_pages', lambda *a: [{'id': 1, 'created_at': '2020-01-01T00:00:00Z'}])
    receipts = [{'date': '2026-11-08', 'papers': [{'url': f'https://pubmed.ncbi.nlm.nih.gov/{n}/'}]} for n in (123, 456)]
    monkeypatch.setattr(history, 'load_delivery_receipt', lambda *a, **kw: receipts)
    saved = []
    monkeypatch.setattr(history, 'persist', lambda entries, stamp: saved.append(entries) or empty_history())
    history.prepare(tmp_path / 'history.json')
    assert len(saved[0]) == 2
