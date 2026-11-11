"""Prepare encrypted-cache-eligible reading inventory; never send email."""
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.curation import curate


def sender_priority_check():
    """Fail closed on unknown sender state; cache checks to bound API traffic."""
    last_checked, last_result = [None], [True]
    def check():
        if last_checked[0] is not None and time.monotonic() - last_checked[0] < 30:
            return last_result[0]
        last_checked[0] = time.monotonic()
        last_result[0] = True
        repository = os.environ.get('GITHUB_REPOSITORY', '')
        token = os.environ.get('GITHUB_TOKEN', '')
        if not repository or not token:
            return True
        try:
            for status in ('in_progress', 'queued'):
                url = f'https://api.github.com/repos/{repository}/actions/runs?status={status}&per_page=100'
                with urlopen(Request(url, headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'}), timeout=3) as response:
                    data = json.load(response)
                runs = data['workflow_runs']
                if data['total_count'] > len(runs):
                    return True
                if any(str(run.get('path', '')).split('@')[0] in ('.github/workflows/main.yml', '.github/workflows/supplement.yml', '.github/workflows/test.yml') for run in runs):
                    return True
            last_result[0] = False
        except Exception:
            pass
        return last_result[0]
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
