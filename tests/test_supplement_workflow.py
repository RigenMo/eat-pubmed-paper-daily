import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def execute(filename, step_name, env, runs=(), now='2026-11-06T22:00:00Z'):
    workflow = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text(encoding='utf-8'))
    steps = [step for job in workflow['jobs'].values() for step in job.get('steps', [])]
    script = next(step['with']['script'] for step in steps if step.get('name') == step_name)
    harness = r'''
const input = JSON.parse(process.argv[1]);
const RealDate = globalThis.Date;
class Date extends RealDate {
  constructor(...args) { super(...(args.length ? args : [input.now])); }
  static now() { return RealDate.parse(input.now); }
}
const context = {repo: {owner: 'owner', repo: 'repo'}, runId: 1};
const dispatches = [];
const listWorkflowRuns = () => {}, listWorkflowRunsForRepo = () => {};
const github = {rest: {actions: {listWorkflowRuns, listWorkflowRunsForRepo,
  createWorkflowDispatch: async params => dispatches.push(params)}},
  paginate: async (method) => method === listWorkflowRuns ? input.runs : input.runs};
Object.assign(process.env, input.env);
(async () => { await (async () => {
''' + script + r'''
})(); console.log(JSON.stringify(dispatches)); })().catch(error => {console.error(error); process.exit(1);});
'''
    result = subprocess.run(['node', '-e', harness, json.dumps(dict(now=now, env=env, runs=runs))], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


MAIN = {'SEND_OUTCOME': 'success', 'QUALITY_PASSED': 'false', 'COMPLETE_CARDS': '8', 'EXPECTED_CARDS': '10'}
RETRY = {'RETRY_SAFE': 'true', 'COMPLETE': 'false'}
RUN = {'id': 1, 'created_at': '2026-11-06T22:00:00Z', 'status': 'in_progress', 'path': '.github/workflows/supplement.yml'}


def main(env=MAIN, runs=(), now='2026-11-06T22:00:00Z'):
    return execute('main.yml', 'Complete missing paper cards', env, runs, now)


def retry(env=RETRY, runs=(RUN,)):
    return execute('supplement.yml', 'Retry remaining shortfall once', env, runs)


def test_initial_dispatch():
    assert main() == [{'owner': 'owner', 'repo': 'repo', 'workflow_id': 'supplement.yml', 'ref': 'main'}]


@pytest.mark.parametrize('change', [
    {'COMPLETE_CARDS': '10'}, {'COMPLETE_CARDS': 'unknown'},
    {'EXPECTED_CARDS': 'unknown'}, {'QUALITY_PASSED': 'unknown'}, {'SEND_OUTCOME': 'failure'},
])
def test_initial_invalid_or_complete(change):
    assert main(MAIN | change) == []


def test_initial_deadline_and_limit():
    assert main(now='2026-11-07T04:00:00Z') == []
    assert main(runs=[RUN, RUN | {'id': 2}]) == []


def test_retry_once():
    assert retry() == [{'owner': 'owner', 'repo': 'repo', 'workflow_id': 'supplement.yml', 'ref': 'main'}]
    assert retry(RETRY | {'RETRY_SAFE': 'false'}) == []
    assert retry(RETRY | {'COMPLETE': 'true'}) == []
    assert retry(runs=[RUN, RUN | {'id': 2}]) == []
