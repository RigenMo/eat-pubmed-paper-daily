"""Build separate no-mail reading reports from current-policy completed cards."""
import hashlib
import json
import os
import queue
import tempfile
import threading
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from zotero_arxiv_daily.card_store import load_ready
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.reading_extensions import extend, synthesize_week


DIRECTORY = Path('.cache/reading-extensions')


def collect(config):
    weekly_enabled = os.environ.get('WEEKLY_SYNTHESIS', '').lower() == 'true'
    deep = {'status': 'unavailable', 'papers': []}
    weekly = {'status': 'unavailable' if weekly_enabled else 'skipped', 'questions': []}
    count = 0
    try:
        executor = Executor(config)
        corpus = executor.filter_corpus(executor.fetch_zotero_corpus())
        if not corpus:
            return count, deep, weekly
        papers = load_ready(config, corpus)
        count = len(papers)
        if not papers:
            deep['status'] = 'insufficient_material'
            if weekly_enabled:
                weekly['status'] = 'insufficient_material'
            return count, deep, weekly
        params = {'model': config.llm.generation_kwargs.model}
        deep['papers'] = extend(papers, executor.openai_client, params)
        deep['status'] = 'success' if any(p['status'] == 'success' for p in deep['papers']) else 'unavailable'
        if weekly_enabled:
            weekly['questions'] = synthesize_week(papers[:20], executor.openai_client, params)
            weekly['status'] = 'success' if weekly['questions'] else 'insufficient_material_or_unavailable'
    except Exception:
        pass
    return count, deep, weekly


def write_report(label, report):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    target = DIRECTORY / (hashlib.sha256(label.encode('ascii')).hexdigest() + '.json')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=DIRECTORY, delete=False) as temp:
        json.dump(report, temp, ensure_ascii=False)
        name = temp.name
    os.replace(name, target)


def run(config):
    results = queue.Queue(maxsize=1)
    def work():
        results.put(collect(config))
    threading.Thread(target=work, daemon=True).start()
    try:
        count, deep, weekly = results.get(timeout=300)
    except queue.Empty:
        count = 0
        deep = {'status': 'unavailable', 'papers': []}
        weekly = {'status': 'unavailable', 'questions': []}
    write_report('deep-reading', deep)
    write_report('weekly-synthesis', weekly)
    return {'sourcecount': count, 'deepcompleted': sum(p.get('status') == 'success' for p in deep['papers']),
            'questions': len(weekly['questions']), 'status': {'deep': deep['status'], 'weekly': weekly['status']}}


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(config: DictConfig):
    logger.remove()
    try:
        print(json.dumps(run(config), ensure_ascii=False), flush=True)
    except Exception:
        print(json.dumps({'sourcecount': 0, 'deepcompleted': 0, 'questions': 0, 'status': 'unavailable'}), flush=True)


if __name__ == '__main__':
    main()
