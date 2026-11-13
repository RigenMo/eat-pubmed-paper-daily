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


@pytest.mark.parametrize('verify_only', [False, True])
def test_full_local_count_is_provisional_and_verify_only_uses_separate_output(tmp_path, monkeypatch, verify_only):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('INVENTORY_VERIFY_ONLY', str(verify_only).lower())
    monkeypatch.setenv('INVENTORY_SUMMARY_FILE', 'outputs/restored.json' if verify_only else 'outputs/inventory-summary.json')
    monkeypatch.setenv('GITHUB_OUTPUT', str(tmp_path / 'step-output'))
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    calls = []
    def curate(config, **kwargs):
        calls.append(kwargs)
        return {'target_met': True, 'stock_count_verified': True, 'stock_after': 20, 'status': 'ok'}
    monkeypatch.setattr(module, 'curate', curate)
    module.main.__wrapped__({})
    output = tmp_path / ('outputs/restored.json' if verify_only else 'outputs/inventory-summary.json')
    summary = json.loads(output.read_text(encoding='utf-8'))
    assert calls[0]['count_only'] is verify_only
    assert summary['stock_count_verified'] is verify_only
    if verify_only:
        assert not (tmp_path / 'step-output').exists()
        assert not (tmp_path / 'outputs/inventory-summary.json').exists()
    else:
        assert summary['durable_stock_verified'] is False
        assert (tmp_path / 'step-output').read_text() == 'target_met=true\n'


@pytest.mark.parametrize('change,accepted', [({}, True), ({'policy': 'different'}, False),
    ({'sent_ledger_digest': 'different'}, False), ({'stock_after': 19}, False),
    ({'stock_count_verified': False}, False), ({'target_met': False}, False),
    ({'status': 'paused_for_sender'}, False), ({'policy': None}, False)])
def test_full_acceptance_requires_matching_independent_restore(tmp_path, change, accepted):
    summary_file, verification_file = tmp_path / 'summary.json', tmp_path / 'restored.json'
    summary = {'policy': 'policy', 'sent_ledger_digest': 'ledger', 'stock_after': 20,
               'stock_count_verified': False, 'target_met': True, 'prepared': 8, 'status': 'ok'}
    restored = dict(summary, stock_count_verified=True, prepared=0)
    restored.update(change)
    summary_file.write_text(json.dumps(summary), encoding='utf-8')
    verification_file.write_text(json.dumps(restored), encoding='utf-8')
    result = module.finalize_restored_stock(summary_file, verification_file)
    assert result['durable_stock_verified'] is result['stock_count_verified'] is result['target_met'] is accepted
    assert result['prepared'] == 8
    if not accepted:
        assert result['status'] == 'stock_unavailable' and result['failure_stage'] == 'cloud_restore_verification'


def test_missing_restore_proof_cannot_leave_full_summary(tmp_path):
    summary_file = tmp_path / 'summary.json'
    summary_file.write_text(json.dumps({'policy': 'policy', 'sent_ledger_digest': 'ledger',
        'stock_after': 20, 'stock_count_verified': False, 'target_met': True}), encoding='utf-8')
    result = module.finalize_restored_stock(summary_file, tmp_path / 'missing.json')
    assert result['stock_count_verified'] is result['target_met'] is False
    assert result['status'] == 'stock_unavailable'


@pytest.mark.parametrize('restored_exists,exit_code', [(False, 1), (True, 0)])
def test_finalize_cli_reports_failed_restore_after_rewriting_summary(tmp_path, monkeypatch, restored_exists, exit_code):
    import runpy
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('INVENTORY_FINALIZE_VERIFICATION', 'true')
    output = tmp_path / 'outputs'
    output.mkdir()
    summary = {'policy': 'policy', 'sent_ledger_digest': 'ledger', 'stock_after': 20,
               'stock_count_verified': False, 'target_met': True, 'status': 'ok'}
    (output / 'inventory-summary.json').write_text(json.dumps(summary), encoding='utf-8')
    if restored_exists:
        (output / 'inventory-restored-summary.json').write_text(json.dumps(dict(summary, stock_count_verified=True)), encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        runpy.run_path(spec.origin, run_name='__main__')
    assert error.value.code == exit_code
    result = json.loads((output / 'inventory-summary.json').read_text(encoding='utf-8'))
    assert result['stock_count_verified'] is restored_exists


def test_workflow_verifies_cloud_restore_before_controller_can_accept_full():
    import yaml
    root = Path(__file__).parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/reading-inventory.yml').read_text(encoding='utf-8'))
    steps = workflow['jobs']['build']['steps']
    produce = next(step for step in steps if step.get('id') == 'produce')
    restore = next(step for step in steps if step.get('id') == 'restore_stock')
    verify = next(step for step in steps if step.get('env', {}).get('INVENTORY_VERIFY_ONLY') == 'true')
    finalize = next(step for step in steps if step.get('env', {}).get('INVENTORY_FINALIZE_VERIFICATION') == 'true')
    upload = next(step for step in steps if step.get('name') == 'Upload aggregate stock status')
    controller = next(step for step in steps if step.get('name') == 'Continue refill or persist backoff')
    cloud_upload = next(step for step in steps if step.get('name') == 'Upload encrypted approved inventory')
    assert steps.index(cloud_upload) < steps.index(restore) < steps.index(verify) < steps.index(finalize) < steps.index(upload) < steps.index(controller)
    assert 'test ! -e "$CARD_STORE_DIR"' in restore['run']
    assert restore['env']['CARD_STORE_DIR'] == verify['env']['CARD_STORE_DIR'] == '.cache/reading-cards-verified'
    assert "steps.restore_stock.outcome == 'success'" in verify['if']
    assert produce['run'].splitlines()[-1] == verify['run'].splitlines()[-1]
    assert verify['run'].splitlines()[0] == 'python3 scripts/sent_paper_history.py prepare'
    assert 'llm.fallback_total_timeout=720 llm.fallback_model_timeout=180' in verify['run']
    for step in (restore, verify, finalize):
        assert "steps.produce.outputs.target_met == 'true'" in step['if']
        assert not any('SMTP' in key for key in step.get('env', {}))
    assert 'outputs/inventory-restored-summary.json' in upload['with']['path']
