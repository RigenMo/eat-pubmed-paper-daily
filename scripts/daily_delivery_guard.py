import io
import json
import os
import zipfile
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler

SHANGHAI = timezone(timedelta(hours=8))


def api_pages(url, token, key):
    page = 1
    while True:
        separator = '&' if '?' in url else '?'
        request = Request(f'{url}{separator}per_page=100&page={page}', headers={
            'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        })
        with urlopen(request, timeout=20) as response:
            items = json.load(response)[key]
        yield from items
        if len(items) < 100:
            break
        page += 1


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected and urlparse(newurl).hostname != 'api.github.com':
            redirected.remove_header('Authorization')
        return redirected


def load_delivery_receipt(repository, token, run_id):
    base = f'https://api.github.com/repos/{repository}/actions'
    artifacts = api_pages(f'{base}/runs/{run_id}/artifacts', token, 'artifacts')
    candidates = [a for a in artifacts if a['name'].startswith('daily-delivery-receipt-') and not a['expired']]
    if not candidates:
        return None
    artifact = max(candidates, key=lambda a: (a['created_at'], a['id']))
    request = Request(f"{base}/artifacts/{artifact['id']}/zip", headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'})
    with build_opener(SafeRedirect()).open(request, timeout=20) as response:
        data = response.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise ValueError('Receipt archive too large')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = [e for e in archive.infolist() if e.filename == 'delivery-receipt.json']
        if len(entries) != 1 or entries[0].file_size > 100 * 1024:
            raise ValueError('Invalid receipt archive')
        receipt = json.loads(archive.read(entries[0]))
    if receipt.get('schema_version') != 1 or receipt.get('run_id') != str(run_id) or receipt.get('mail_sent') is not True:
        raise ValueError('Invalid receipt identity or status')
    if receipt.get('run_attempt') is not None and not isinstance(receipt['run_attempt'], str):
        raise ValueError('Invalid receipt attempt')
    recorded = datetime.fromisoformat(receipt['recorded_at_utc'].replace('Z', '+00:00'))
    if recorded.tzinfo is None or recorded.utcoffset() != timedelta(0) or recorded.astimezone(SHANGHAI).date().isoformat() != receipt['date']:
        raise ValueError('Invalid receipt date')
    for field in ('paper_count', 'complete_cards', 'expected_cards'):
        if type(receipt.get(field)) is not int or receipt[field] < 0:
            raise ValueError('Invalid receipt count')
    count, complete, expected = (receipt[k] for k in ('paper_count', 'complete_cards', 'expected_cards'))
    if not expected or complete > count or type(receipt.get('quality_passed')) is not bool or receipt['quality_passed'] != (count >= expected and complete == count):
        raise ValueError('Invalid receipt quality')
    if not isinstance(receipt.get('papers'), list) or len(receipt['papers']) != count:
        raise ValueError('Invalid receipt papers')
    if any(not isinstance(p, dict) or not all(k in p for k in ('url', 'journal', 'publication_date', 'jif')) for p in receipt['papers']):
        raise ValueError('Invalid receipt paper')
    return receipt


def delivery_status(repository, token, current_run, now=None, notice_kind="failure", progress=None):
    now = now or datetime.now(SHANGHAI)
    today = now.astimezone(SHANGHAI).date()
    start = datetime.combine(today - timedelta(days=1), datetime.min.time(), SHANGHAI).astimezone(timezone.utc)
    query = urlencode({'created': '>=' + start.strftime('%Y-%m-%dT%H:%M:%SZ')})
    base = f'https://api.github.com/repos/{repository}/actions'
    delivered = notified = unknown = False
    urls = set()
    for workflow in ('main.yml', 'supplement.yml', 'keep-alive.yml'):
        try:
            runs = list(api_pages(f'{base}/workflows/{workflow}/runs?{query}', token, 'workflow_runs'))
        except HTTPError as error:
            if workflow == 'supplement.yml' and error.code == 404:
                continue
            raise
        if workflow != 'supplement.yml' and current_run is not None and all(str(r['id']) != str(current_run) for r in runs):
            runs.append({'id': current_run})
        for run in runs:
            run_id = str(run['id'])
            receipt = None
            if workflow in ('main.yml', 'supplement.yml'):
                receipt = load_delivery_receipt(repository, token, run_id)
                if receipt and receipt['date'] == today.isoformat():
                    delivered |= receipt['paper_count'] > 0
                    if receipt['paper_count'] != receipt['complete_cards']:
                        unknown = True
                    else:
                        for paper in receipt['papers']:
                            url = paper['url']
                            if not isinstance(url, str) or not url.strip():
                                unknown = True
                            else:
                                urls.add(url.strip())
            for job in api_pages(f'{base}/runs/{run_id}/jobs?filter=all', token, 'jobs'):
                for step in (job.get('steps') or []):
                    if step.get('conclusion') != 'success':
                        continue
                    is_delivery = workflow == 'main.yml' and job['name'] == 'calculate-and-send' and step['name'] == 'Run script'
                    is_supplement = workflow == 'supplement.yml' and job['name'] == 'supplement' and step['name'] == 'Send missing cards'
                    is_notice = job['name'] == 'notify-on-failure' and ((workflow == 'main.yml' and step['name'] == 'Send failure notice') or (workflow == 'keep-alive.yml' and step['name'] == 'Send supervisor failure notice'))
                    if notice_kind == 'morning':
                        is_notice = workflow == 'keep-alive.yml' and job['name'] == 'morning-deadline' and step['name'] == 'Send morning deadline notice'
                    if not (is_delivery or is_supplement or is_notice):
                        continue
                    completed_at = step.get('completed_at')
                    if not isinstance(completed_at, str):
                        raise ValueError('Successful send step has no completion time')
                    completed = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                    if completed.tzinfo is None:
                        raise ValueError('Successful send step has no timezone')
                    if completed.astimezone(SHANGHAI).date() == today:
                        if is_delivery or is_supplement:
                            attempt = job.get('run_attempt')
                            receipt_matches = receipt and (attempt is None or receipt.get('run_attempt') is None or str(attempt) == receipt['run_attempt'])
                            if not receipt_matches:
                                unknown = True
                                delivered |= is_delivery
                            elif receipt['date'] == today.isoformat() and receipt['paper_count'] > 0:
                                delivered = True
                        notified |= is_notice
    if progress is not None:
        progress.update(paper_count=len(urls), day_complete=len(urls) >= 10, delivery_unknown=unknown)
    return delivered, notified


if __name__ == '__main__':
    progress = {}
    delivered, notified = delivery_status(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN'], os.environ['GITHUB_RUN_ID'], notice_kind=os.environ.get('NOTICE_KIND', 'failure'), progress=progress)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write(f'delivered={str(delivered).lower()}\nnotified={str(notified).lower()}\n')
        for key, value in progress.items():
            output.write(f'{key}={str(value).lower()}\n')
    print(f'Delivery status: delivered={delivered}, notified={notified}')


