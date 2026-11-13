"""Prepare encrypted-cache-eligible reading inventory; never send email."""
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.curation import curate


def read_sender_runs(repository, token, workflow, status):
    url = f'https://api.github.com/repos/{repository}/actions/workflows/{workflow}/runs?status={status}&per_page=1'
    for attempt in range(2):
        try:
            request = Request(url, headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                                            'User-Agent': 'eat-paper-inventory', 'X-GitHub-Api-Version': '2022-11-28'})
            with urlopen(request, timeout=5) as response:
                return json.load(response)
        except (HTTPError, URLError, OSError) as error:
            code = getattr(error, 'code', None)
            transient = not isinstance(error, HTTPError) or code == 429 or (type(code) is int and code >= 500)
            if transient and attempt == 0:
                continue
            status_code = code if type(code) is int else 'unknown'
            reason = 'http_error' if isinstance(error, HTTPError) else 'network_or_timeout'
            print(f'Stock sender-state check unavailable: endpoint=sender_status, stage=request, reason={reason}, status={status_code}', flush=True)
            raise


def sender_priority_check():
    """Fail closed on unknown sender state; cache checks to bound API traffic."""
    last_checked, last_result = [None], [True]
    def check():
        if last_checked[0] is not None and time.monotonic() - last_checked[0] < 30:
            return last_result[0]
        last_checked[0] = time.monotonic()
        last_result[0] = True
        check.reason = 'sender_status_unavailable'
        repository = os.environ.get('GITHUB_REPOSITORY', '')
        token = os.environ.get('GITHUB_TOKEN', '')
        if not repository or not token:
            return True
        try:
            for workflow in ('main.yml', 'supplement.yml', 'test.yml'):
                for status in ('in_progress', 'queued'):
                    data = read_sender_runs(repository, token, workflow, status)
                    runs = data.get('workflow_runs') if isinstance(data, dict) else None
                    count = data.get('total_count') if isinstance(data, dict) else None
                    if (not isinstance(runs, list) or type(count) is not int or count < 0 or len(runs) > 1 or
                            count < len(runs) or (count == 0) != (len(runs) == 0) or
                            any(not isinstance(run, dict) or not isinstance(run.get('path'), str) or
                                run['path'].split('@')[0] != '.github/workflows/' + workflow or run.get('status') != status for run in runs)):
                        print('Stock sender-state check unavailable: endpoint=sender_status, stage=validate, reason=invalid_sender_response, status=unknown', flush=True)
                        return True
                    if count > 0:
                        check.reason = 'sender_active'
                        return True
            last_result[0] = False
            check.reason = 'sender_idle'
        except (HTTPError, URLError, OSError):
            pass  # The bounded read already recorded a safe diagnostic.
        except Exception:
            print('Stock sender-state check unavailable: endpoint=sender_status, stage=decode, reason=invalid_response, status=unknown', flush=True)
        return last_result[0]
    check.reason = 'sender_status_unavailable'
    return check


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(config: DictConfig):
    seconds = int(os.environ.get('INVENTORY_BUILD_SECONDS', '2700'))
    max_new = int(os.environ.get('INVENTORY_MAX_NEW_CARDS', '12'))
    if not 1 <= seconds <= 2700 or not 1 <= max_new <= 12:
        raise ValueError('Invalid inventory duration or card limit')
    priority = sender_priority_check() if os.environ.get('GITHUB_ACTIONS') == 'true' else None
    result = curate(config, deadline=time.monotonic() + seconds, max_new=max_new, should_yield=priority)
    output = Path('outputs/inventory-summary.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
