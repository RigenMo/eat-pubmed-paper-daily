import json
import subprocess
from pathlib import Path

import pytest


WORKER = (Path(__file__).resolve().parents[1] / 'scheduler' / 'worker.mjs').as_uri()


def run_worker(scenario):
    script = r'''
import { check } from WORKER;
const scenario = SCENARIO;
const calls = [];
const secret = 'example-secret-not-for-output';
const fetcher = async (url, options) => {
  calls.push({url, method: options.method || 'GET', body: options.body});
  if (scenario === 'network') throw new Error(secret);
  if (scenario === 'http_error') return new Response(secret, {status: 403});
  if (url.includes('/runs?')) {
    const active = scenario === 'active' && url.includes('queued');
    return Response.json({total_count: active ? 1 : 0, workflow_runs: active ? [{path: '.github/workflows/main.yml'}] : []});
  }
  if (url.endsWith('/keep-alive.yml')) return Response.json({state: scenario === 'disabled' ? 'disabled_inactivity' : 'active'});
  return new Response(null, {status: 204});
};
try {
  const result = await check({REPOSITORY: 'RigenMo/eat-pubmed-paper-daily', GH_ACTIONS_TOKEN: scenario === 'missing' ? '' : secret}, fetcher);
  console.log(JSON.stringify({result, calls}));
} catch (error) {
  console.log(JSON.stringify({error: String(error), calls}));
}
'''.replace('WORKER', json.dumps(WORKER)).replace('SCENARIO', json.dumps(scenario))
    proc = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    assert 'example-secret-not-for-output' not in proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def test_active_sender_skips_dispatch():
    result = run_worker('active')
    assert result['result'] == 'active_sender'
    assert len(result['calls']) == 1


def test_dispatches_only_existing_supervisor():
    result = run_worker('normal')
    assert result['result'] == 'requested_supervisor'
    assert len(result['calls']) == 4
    call = result['calls'][-1]
    assert call['url'].endswith('/workflows/keep-alive.yml/dispatches')
    assert call['method'] == 'POST'
    assert json.loads(call['body']) == {'ref': 'main', 'inputs': {'recover': 'true'}}


def test_disabled_supervisor_is_enabled_then_waits():
    result = run_worker('disabled')
    assert result['result'] == 'enabled_supervisor'
    assert len(result['calls']) == 4
    assert result['calls'][-1]['url'].endswith('/workflows/keep-alive.yml/enable')
    assert result['calls'][-1]['method'] == 'PUT'


def test_missing_secret_never_calls_github():
    result = run_worker('missing')
    assert result['calls'] == []
    assert 'WATCHDOG_CONFIGURATION_ERROR' in result['error']


@pytest.mark.parametrize('scenario', ['http_error', 'network'])
def test_failures_are_safe_and_not_success(scenario):
    result = run_worker(scenario)
    assert 'WATCHDOG_' in result['error']
    assert len(result['calls']) == 1
