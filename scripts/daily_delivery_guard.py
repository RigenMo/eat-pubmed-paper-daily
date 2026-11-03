import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SHANGHAI = timezone(timedelta(hours=8))


def api_pages(url, token, key):
    page = 1
    while True:
        separator = '&' if '?' in url else '?'
        request = Request(f'{url}{separator}per_page=100&page={page}', headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
        })
        with urlopen(request, timeout=20) as response:
            items = json.load(response)[key]
        yield from items
        if len(items) < 100:
            break
        page += 1


def delivery_status(repository, token, current_run, now=None):
    now = now or datetime.now(SHANGHAI)
    today = now.astimezone(SHANGHAI).date()
    start = datetime.combine(today - timedelta(days=1), datetime.min.time(), SHANGHAI).astimezone(timezone.utc)
    query = urlencode({'status': 'completed', 'created': '>=' + start.strftime('%Y-%m-%dT%H:%M:%SZ')})
    base = f'https://api.github.com/repos/{repository}/actions'
    runs = api_pages(f'{base}/workflows/main.yml/runs?{query}', token, 'workflow_runs')
    delivered = notified = False
    for run in runs:
        if str(run['id']) == str(current_run):
            continue
        for job in api_pages(f"{base}/runs/{run['id']}/jobs?filter=all", token, 'jobs'):
            for step in job.get('steps', []):
                if step.get('conclusion') != 'success':
                    continue
                is_delivery = job['name'] == 'calculate-and-send' and step['name'] == 'Run script'
                is_notice = job['name'] == 'notify-on-failure' and step['name'] == 'Send failure notice'
                if not (is_delivery or is_notice):
                    continue
                completed_at = step.get('completed_at')
                if not isinstance(completed_at, str):
                    raise ValueError('Successful send step has no completion time')
                completed = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                if completed.tzinfo is None:
                    raise ValueError('Successful send step has no timezone')
                if completed.astimezone(SHANGHAI).date() != today:
                    continue
                if is_delivery:
                    delivered = True
                if is_notice:
                    notified = True
    return delivered, notified


if __name__ == '__main__':
    delivered, notified = delivery_status(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN'], os.environ['GITHUB_RUN_ID'])
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write(f'delivered={str(delivered).lower()}\nnotified={str(notified).lower()}\n')
    print(f'Delivery status: delivered={delivered}, notified={notified}')

