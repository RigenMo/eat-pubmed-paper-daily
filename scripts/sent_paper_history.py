import base64
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen, build_opener

from scripts.daily_delivery_guard import SafeRedirect, api_pages, load_delivery_receipt
from zotero_arxiv_daily.paper_identity import load_history, paper_keys

BRANCH = 'codex/sent-paper-history'
REMOTE = '.github/sent-paper-history.json'


def request(method, path, payload=None):
    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}"
    headers = {'Authorization': f"Bearer {os.environ['GITHUB_TOKEN']}", 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
    data = json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(url, data=data, headers=headers, method=method), timeout=20) as response:
        return json.load(response)


def validated(data):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'history.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        result = load_history(path, required=True)
    stamp = datetime.fromisoformat(result['reconciled_at'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0) or result['retention_days'] != 180 or set(result) != {'version', 'entries', 'reconciled_at', 'retention_days'}:
        raise ValueError('Invalid history metadata')
    return result


def read_remote():
    try:
        item = request('GET', f'contents/{REMOTE}?ref={quote(BRANCH, safe="")}')
    except HTTPError as error:
        if error.code != 404:
            raise
        try:
            request('GET', f'git/ref/heads/{BRANCH}')
        except HTTPError as branch_error:
            if branch_error.code != 404:
                raise
            return None, None
        raise ValueError('History branch exists but history file is missing')
    content = base64.b64decode(item['content'], validate=False)
    return validated(json.loads(content)), item['sha']


def persist(additions, reconciled_at=None):
    for attempt in range(4):
        current, sha = read_remote()
        if current is None:
            repository = request('GET', '')
            head = request('GET', f"git/ref/heads/{repository['default_branch']}")
            try:
                request('POST', 'git/refs', {'ref': f'refs/heads/{BRANCH}', 'sha': head['object']['sha']})
            except HTTPError as error:
                if error.code == 422 and attempt < 3:
                    continue
                raise
        current = current or {'version': 1, 'entries': {}, 'reconciled_at': '1970-01-01T00:00:00+00:00', 'retention_days': 180}
        for key, day in additions.items():
            current['entries'][key] = max(day, current['entries'].get(key, day))
        if reconciled_at:
            current['reconciled_at'] = max(current['reconciled_at'], reconciled_at)
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=180)).isoformat()
        current['entries'] = {k: v for k, v in current['entries'].items() if v >= cutoff}
        payload = {'message': 'Update sent paper history', 'branch': BRANCH, 'content': base64.b64encode(json.dumps(current, sort_keys=True).encode()).decode()}
        if sha:
            payload['sha'] = sha
        try:
            request('PUT', f'contents/{REMOTE}', payload)
            return current
        except HTTPError as error:
            if error.code not in (409, 422) or attempt == 3:
                raise


def receipt_entries(receipt):
    result = {}
    for paper in receipt['papers']:
        keys = paper_keys(paper)
        if not keys:
            raise ValueError('Sent paper has no identifiable key')
        for key in keys:
            result[key] = receipt['date']
    return result


def verify_missing_receipts(repo, token, run_id, workflow, receipts):
    base = f'https://api.github.com/repos/{repo}/actions'
    jobs = api_pages(f'{base}/runs/{run_id}/jobs?filter=all', token, 'jobs')
    attempts = {r['run_attempt'] for r in receipts}
    job_name = 'supplement' if workflow == 'supplement.yml' else 'calculate-and-send'
    step_name = 'Send missing cards' if workflow == 'supplement.yml' else 'Run script'
    for job in jobs:
        if job.get('name') != job_name or str(job.get('run_attempt')) in attempts:
            continue
        steps = job.get('steps') or []
        send = next ( (step for step in steps if step.get('name') == step_name), None )
        if not send or send.get('conclusion') == 'skipped':
            continue
        if send.get('conclusion') is None:
            raise RuntimeError('Send still running without confirmed receipt')
        request = Request(f"{base}/jobs/{job['id']}/logs", headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'})
        with build_opener(SafeRedirect()).open(request, timeout=20) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError('Job log too large')
        log = data.decode('utf-8', errors='replace')
        saved = any(step.get('name') == 'Save sent paper history' and step.get('conclusion') == 'success' for step in steps)
        if saved and 'SENT_PAPER_HISTORY_SAVED' in log:
            continue
        if send.get('conclusion') == 'success' or 'Sending email...' in log or 'SUPPLEMENT_SMTP_STARTED' in log:
            raise RuntimeError('Possible SMTP delivery without receipt or saved history')


def prepare(path):
    current, _ = read_remote()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=180)
    if current:
        start = max(start, datetime.fromisoformat(current['reconciled_at'].replace('Z', '+00:00')) - timedelta(days=2))
    query = urlencode({'created': '>=' + start.strftime('%Y-%m-%dT%H:%M:%SZ')})
    repo, token = os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN']
    additions = {}
    missing = 0
    confirmed = 0
    scanned = 0
    for workflow in ('main.yml', 'supplement.yml', 'test.yml'):
        url = f'https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?{query}'
        try:
            runs = list(api_pages(url, token, 'workflow_runs'))
        except HTTPError as error:
            if error.code == 404:
                continue
            raise
        for run in runs:
            if str(run['id']) == os.environ.get('GITHUB_RUN_ID'):
                continue
            scanned += 1
            receipts = load_delivery_receipt(repo, token, str(run['id']), all_attempts=True)
            if not receipts:
                missing += 1
            for receipt in receipts:
                confirmed += 1
                for key, day in receipt_entries(receipt).items():
                    additions[key] = max(day, additions.get(key, day))
            created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
            boundary = now - timedelta(days=2)
            if current:
                boundary = min(boundary, datetime.fromisoformat(current['reconciled_at'].replace('Z', '+00:00')))
            if created >= boundary:
                verify_missing_receipts(repo, token, str(run['id']), workflow, receipts)
    print(f'Coverage start: {start.isoformat()}; scanned runs: {scanned}; confirmed receipts: {confirmed}; unavailable receipts: {missing}')
    result = persist(additions, now.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as temp:
        name = temp.name
        json.dump(result, temp)
    os.replace(name, path)


def save():
    path = Path('outputs/delivery-receipt.json')
    if not path.exists():
        return
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('schema_version') != 1 or receipt.get('mail_sent') is not True or receipt.get('run_id') != os.environ['GITHUB_RUN_ID'] or receipt.get('run_attempt') != os.environ.get('GITHUB_RUN_ATTEMPT'):
        raise ValueError('Invalid local receipt identity')
    recorded = datetime.fromisoformat(receipt['recorded_at_utc'].replace('Z', '+00:00'))
    if recorded.utcoffset() != timedelta(0) or recorded.astimezone(timezone(timedelta(hours=8))).date().isoformat() != receipt['date']:
        raise ValueError('Invalid local receipt date')
    count, complete, expected = (receipt[k] for k in ('paper_count', 'complete_cards', 'expected_cards'))
    if any(type(n) is not int or n < 0 for n in (count, complete, expected)) or not expected or complete > count or type(receipt.get('quality_passed')) is not bool or receipt['quality_passed'] != (count >= expected and complete == count) or not isinstance(receipt.get('papers'), list) or len(receipt['papers']) != count or any(not isinstance(p, dict) or not all(k in p for k in ('url', 'journal', 'publication_date', 'jif')) for p in receipt['papers']):
        raise ValueError('Invalid local receipt content')
    persist(receipt_entries(receipt))
    print('SENT_PAPER_HISTORY_SAVED')


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare(Path(os.environ.get('SENT_PAPER_HISTORY_FILE', '.cache/sent-paper-history.json')))
    elif sys.argv[1:] == ['save']:
        save()
    else:
        raise SystemExit('Usage: sent_paper_history.py prepare|save')
