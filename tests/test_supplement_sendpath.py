import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location('supplement_sendpath', SCRIPTS / 'supplement_digest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.remove(str(SCRIPTS))


class FakePaper:
    def __init__(self, identifier, journal):
        self.url = f'https://pubmed.ncbi.nlm.nih.gov/{identifier}/'
        self.journal = journal
        self.publication_date = '2026-11-01'
        self.journal_metrics = {'jif': 12}
        self.reading_card = None
        self.abstract_zh = None
        self.reader_status = None

    def generate_tldr(self, *args):
        if self.url.endswith('/9/'):
            self.reader_status = 'failed'
            return
        self.reading_card = 'complete card'
        self.abstract_zh = '中文摘要'
        self.reader_status = 'success'

    def generate_title_translation(self, *args):
        pass


def setup_send(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    history = tmp_path / 'sent-paper-history.json'
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(history))
    module.record_sent(history, [], '2026-11-01')
    for key, value in {'GITHUB_TOKEN': 'test', 'GITHUB_REPOSITORY': 'owner/repo',
                       'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1',
                       'GITHUB_OUTPUT': str(tmp_path / 'github-output')}.items():
        monkeypatch.setenv(key, value)
    journals = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'B', 'A', 'E', 'F']
    papers = {str(i): FakePaper(i, journal) for i, journal in enumerate(journals, 1)}
    receipt = {'papers': [{'url': papers[str(i)].url, 'journal': papers[str(i)].journal,
                           'publication_date': papers[str(i)].publication_date, 'jif': 12}
                          for i in range(1, 9)]}
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt], [42], 1))
    monkeypatch.setattr(module, 'read_candidates', lambda *args: ['1', '9', '10', '11', '12'])
    retriever = SimpleNamespace(_fetch_details=lambda ids: [papers[i] for i in ids],
                                convert_to_paper=lambda record: record)
    executor = SimpleNamespace(retrievers={'pubmed': retriever}, openai_client=object(),
                               reranker=SimpleNamespace(rerank=lambda eligible, *a, **k: eligible),
                               fetch_zotero_corpus=lambda: ['reference'], filter_corpus=lambda x: x,
                               generate_email_summary=lambda selected: 'summary')
    monkeypatch.setattr(module, 'Executor', lambda config: executor)
    monkeypatch.setattr(module, '_publication_in_window', lambda *args: True)
    monkeypatch.setattr(module, 'enrich_journal_metrics', lambda eligible, *args: eligible)
    monkeypatch.setattr(module, 'effective_focus_mode', lambda settings: 'test')
    monkeypatch.setattr(module, 'focus_evidence', lambda *args: None)
    monkeypatch.setattr(module, 'verify_recommendations', lambda eligible, *a, **k: eligible)
    monkeypatch.setattr(module, 'assess_innovation', lambda *args: None)
    monkeypatch.setattr(module, 'order_candidates', lambda eligible, mode: eligible)
    sent = []
    monkeypatch.setattr(module, 'render_email', lambda selected, **kwargs: list(selected))
    monkeypatch.setattr(module, 'send_email', lambda config, html, **kwargs: sent.append((html, kwargs)))
    config = OmegaConf.create({'recommendations': {'max_per_journal': 3}, 'journals': {},
                               'executor': {}, 'llm': {}})
    return config, receipt, papers, sent


def test_sends_only_two_new_complete_cards_and_records_day(monkeypatch, tmp_path):
    config, previous, papers, sent = setup_send(monkeypatch, tmp_path)
    assert module.supplement(config) == 2
    assert len(sent) == 1
    selected = sent[0][0]
    assert [p.url for p in selected] == [papers['11'].url, papers['12'].url]
    assert not ({p.url for p in selected} & {p['url'] for p in previous['papers']})
    counts = Counter(p['journal'].casefold() for p in previous['papers'])
    counts.update(p.journal.casefold() for p in selected)
    assert max(counts.values()) <= 3
    assert papers['9'].reader_status == 'failed'
    assert papers['10'].reader_status is None  # journal A is already at its daily cap

    receipt = json.loads((tmp_path / 'outputs/delivery-receipt.json').read_text(encoding='utf-8'))
    assert receipt['mail_sent'] is True and receipt['quality_passed'] is True
    assert receipt['is_supplement'] is True
    assert (receipt['paper_count'], receipt['complete_cards'], receipt['expected_cards']) == (2, 2, 2)
    assert (receipt['day_complete_cards'], receipt['day_expected_cards']) == (10, 10)
    assert [p['url'] for p in receipt['papers']] == [p.url for p in selected]
    summary = json.loads((tmp_path / 'outputs/supplement-summary.json').read_text(encoding='utf-8'))
    assert summary['complete'] is True and summary['sent'] == 2 and summary['cumulative'] == 10
    assert summary['retry_safe'] is True and summary['remaining'] == 0
    output = (tmp_path / 'github-output').read_text(encoding='utf-8')
    assert 'complete=true' in output and 'cumulative=10' in output and 'retry_safe=true' in output


def test_history_credential_failure_never_sends(monkeypatch, tmp_path):
    config, _, _, sent = setup_send(monkeypatch, tmp_path)
    def fail(*args):
        raise PermissionError('invalid credentials')
    monkeypatch.setattr(module, 'today_history', fail)
    with pytest.raises(PermissionError, match='invalid credentials'):
        module.supplement(config)
    assert not sent
    assert not (tmp_path / 'outputs/delivery-receipt.json').exists()


def test_partial_cache_is_sent_when_new_candidate_processing_fails(monkeypatch, tmp_path):
    config, _, papers, sent = setup_send(monkeypatch, tmp_path)
    config.recommendations.card_cache_enabled = True
    config.zotero = {'include_path': None, 'ignore_path': None}
    cached = papers['11']
    cached.reading_card, cached.abstract_zh, cached.reader_status = {'complete': True}, '摘要', 'success'

    class FailingExecutor:
        def __init__(self, config):
            raise RuntimeError('new candidate service failed')

        def fetch_zotero_corpus(self):
            return ['reference']

        def filter_corpus(self, corpus):
            return corpus

    monkeypatch.setattr(module, 'Executor', FailingExecutor)
    monkeypatch.setattr(module, 'load_ready', lambda *args: [cached])
    retriever = SimpleNamespace(_fetch_details=lambda ids: [papers[i] for i in ids],
                                convert_to_paper=lambda record: record)
    monkeypatch.setattr(module, 'PubMedRetriever', lambda config: retriever)
    assert module.supplement(config) == 1
    assert sent[0][0] == [cached]
