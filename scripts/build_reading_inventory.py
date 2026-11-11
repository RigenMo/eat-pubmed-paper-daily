"""Prepare encrypted-cache-eligible reading inventory; never send email."""
import json
import os
from pathlib import Path
import sys
import time

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.curation import curate


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(config: DictConfig):
    seconds = int(os.environ.get('INVENTORY_BUILD_SECONDS', '2700'))
    max_new = int(os.environ.get('INVENTORY_MAX_NEW_CARDS', '12'))
    if not 1 <= seconds <= 2700 or not 1 <= max_new <= 12:
        raise ValueError('Invalid inventory duration or card limit')
    result = curate(config, deadline=time.monotonic() + seconds, max_new=max_new)
    output = Path('outputs/inventory-summary.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
