import json
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/keep-alive.yml'


def run_prepare(now, times=(), delivered=False, active=False):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    steps = workflow['jobs']['check-daily']['steps']
    script = next(step['with']['script'] for step in steps if step['name'] == 'Prepare recovery')
    harness = r'''
const input = JSON.parse(process.argv[1]);
const RealDate = globalThis.Date;
class FakeDate extends RealDate { static now() { return RealDate.parse(input.now); } }
const Date = FakeDate;
const context = {repo: {owner: 'owner', repo: 'repo'}, runId: 999};
const outputs = {};
const core = {info: () => {}, setOutput: (key, value) => { outputs[key] = value; }};
const listWorkflowRuns = () => {}, listJobsForWorkflowRun = () => {};
const github = {rest: {actions: {listWorkflowRuns, listJobsForWorkflowRun,
  getWorkflow: async () => ({data: {state: 'active'}})},
  repos: {get: async () => ({data: {default_branch: 'main'}})}},
  paginate: async (method, params) => {
    if (method === listJobsForWorkflowRun) return [{steps: [{name: 'Dispatch missing daily email', conclusion: 'success', completed_at: input.times[params.run_id - 1]}]}];
    if (params.workflow_id === 'main.yml') return input.active ? [{status: 'in_progress'}] : [];
    return input.times.map((_, index) => ({id: index + 1}));
  }};
process.env.DELIVERED = input.delivered ? 'true' : 'false';
process.env.EVENT_NAME = 'workflow_dispatch';
process.env.RECOVER = 'true';
(async () => { await (async () => {
''' + script + r'''
})(); console.log(JSON.stringify(outputs)); })().catch(error => { console.error(error); process.exit(1); });
'''
    result = subprocess.run(['node', '-e', harness, json.dumps(dict(now=now, times=times, delivered=delivered, active=active))], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)['needs_dispatch']


def test_delivered_and_active_skip():
    assert run_prepare('2026-11-07T01:00:00Z', delivered=True) == 'false'
    assert run_prepare('2026-11-07T01:00:00Z', active=True) == 'false'


def test_daily_limit_and_shanghai_date():
    # Previous UTC date, but same Shanghai date as the current run.
    times = [f'2026-11-06T{hour}:00:00Z' for hour in ('16', '17', '18', '19', '20')]
    assert run_prepare('2026-11-07T01:00:00Z', times) == 'true'
    assert run_prepare('2026-11-07T01:00:00Z', times + ['2026-11-06T21:00:00Z']) == 'false'
    assert run_prepare('2026-11-07T01:00:00Z', times + ['2026-11-06T15:59:59Z']) == 'true'


@pytest.mark.parametrize(('now', 'expected'), [('2026-11-07T00:29:59Z', 'false'), ('2026-11-07T00:30:00Z', 'true')])
def test_cooldown(now, expected):
    assert run_prepare(now, ['2026-11-07T00:00:00Z']) == expected


@pytest.mark.parametrize(('now', 'expected'), [('2026-11-07T01:59:00Z', 'true'), ('2026-11-07T02:00:00Z', 'false')])
def test_deadline(now, expected):
    assert run_prepare(now) == expected

