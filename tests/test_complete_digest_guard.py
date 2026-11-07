from datetime import datetime
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper


def test_automatic_empty_results_do_not_send_a_blank_digest(config, monkeypatch):
    with open_dict(config):
        config.search.mode = 5
        config.zotero.user_id = 'test'
        config.zotero.api_key = 'test'
        config.llm.api.key = 'test'
        config.llm.api.base_url = 'https://example.test/v1'
        config.source.mix_mode = 1
        config.recommendations.enabled = True
        config.recommendations.require_cards = True
        config.executor.send_empty = True
        config.journals.enabled = False
    executor = Executor(config)
    corpus = [CorpusPaper(title='Multimorbidity study', abstract='Longitudinal disease progression',
                          added_date=datetime(2026, 11, 7), paths=['test'])]
    monkeypatch.setattr(executor, 'fetch_zotero_corpus', lambda: corpus)
    monkeypatch.setattr(executor, 'filter_corpus', lambda value: value)
    monkeypatch.setattr('zotero_arxiv_daily.executor.build_zotero_queries', lambda *a, **k: ['test'])
    executor.retrievers = {'pubmed': SimpleNamespace(set_pubmed_queries=lambda value: None,
                                                    retrieve_papers=lambda: [])}
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **k: pytest.fail('blank report sent'))
    with pytest.raises(RuntimeError, match='No qualifying complete papers'):
        executor.run(send_mail=True)
    assert executor.run(send_mail=False) == 0
