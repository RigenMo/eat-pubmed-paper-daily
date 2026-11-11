import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from zotero_arxiv_daily.card_store import load_ready, save_ready
from zotero_arxiv_daily.protocol import Paper


def test_public_cache_round_trip_and_revalidation(tmp_path, monkeypatch):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True, 'max_papers': 10}})
    corpus = [SimpleNamespace(title='PRIVATE LIBRARY TITLE', abstract='PRIVATE LIBRARY ABSTRACT')]
    sentence = '这项研究基于所提供的摘要材料进行审慎分析并报告相关结果。' * 2
    card = {
        'essence': [sentence, sentence],
        'design': {'objects_data': sentence, 'task_definitions': sentence,
                   'implementation': sentence, 'comparison_validation_scope': sentence},
        'evidence': [{'claim': '结果', 'comparison': '比较', 'result': '观察', 'locator': '摘要段落1'}],
        'innovation': [sentence, sentence], 'criticism': [sentence, sentence, sentence],
        'self_test': '如何理解这项研究的主要发现？',
    }
    today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    paper = Paper(source='pubmed', title='Public paper', authors=['Author'], abstract='Public abstract.',
                  url='https://pubmed.ncbi.nlm.nih.gov/12345/', journal='Journal',
                  publication_date=today, journal_metrics={'jif': 12}, abstract_zh='中文摘要',
                  reading_card=card, material_scope='仅据摘要（正文与补充材料未提供）', reader_status='success',
                  recommendation_matches=[{'quote': 'PRIVATE QUOTE'}])
    save_ready(paper, config, corpus)
    paths = list(tmp_path.glob('*.json'))
    assert len(paths) == 1
    text = paths[0].read_text(encoding='utf-8')
    assert 'PRIVATE' not in text
    assert len(load_ready(config, corpus)) == 1
    assert not load_ready(config, [SimpleNamespace(title='changed', abstract='changed')])
    payload = json.loads(text)
    payload['paper']['reading_card']['evidence'][0]['locator'] = '摘要段落99'
    paths[0].write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    assert load_ready(config, corpus) == []


def test_full_cached_main_skips_retrieval_and_llm(config, monkeypatch, tmp_path):
    from tests.test_executor import _delivery_executor
    from tests.canned_responses import make_sample_paper

    config.recommendations.enabled = True
    config.recommendations.max_papers = 2
    config.recommendations.max_per_journal = 2
    config.recommendations.exclude_sent = False
    config.recommendations.require_cards = True
    config.executor.max_paper_num = 2
    config.executor.debug = False
    config.journals.enabled = False
    papers = [make_sample_paper(title=f'Cached {i}') for i in range(2)]
    for i, paper in enumerate(papers):
        paper.url = f'https://pubmed.ncbi.nlm.nih.gov/{i + 1}/'
        paper.reading_card = {'complete': True}
        paper.abstract_zh = '摘要'
        paper.innovation_assessment = {}
    executor = _delivery_executor(config, [])
    executor._search_mode = 2
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda corpus: corpus
    executor.retrievers = {'pubmed': SimpleNamespace(retrieve_papers=lambda: (_ for _ in ()).throw(AssertionError('retrieval')))}
    monkeypatch.setattr('zotero_arxiv_daily.executor.load_ready', lambda *args: papers)
    monkeypatch.setattr('zotero_arxiv_daily.executor.focus_evidence', lambda *args: None)
    monkeypatch.setattr('zotero_arxiv_daily.executor.render_email', lambda papers, **kwargs: 'rendered')
    executor.generate_email_summary = lambda papers: (_ for _ in ()).throw(AssertionError('LLM'))
    monkeypatch.chdir(tmp_path)
    assert executor.run(send_mail=False) == 2


@pytest.mark.parametrize('failure', ['query', 'metrics'])
def test_partial_cached_main_survives_new_candidate_failure(config, monkeypatch, tmp_path, failure):
    from tests.test_executor import _delivery_executor
    from tests.canned_responses import make_sample_paper

    config.recommendations.enabled = True
    config.recommendations.max_papers = 2
    config.recommendations.max_per_journal = 2
    config.recommendations.exclude_sent = False
    config.recommendations.require_cards = True
    config.executor.max_paper_num = 2
    config.executor.debug = False
    config.journals.enabled = False
    cached = make_sample_paper(title='Cached')
    cached.url = 'https://pubmed.ncbi.nlm.nih.gov/1/'
    cached.reading_card, cached.abstract_zh, cached.innovation_assessment = {'complete': True}, '摘要', {}
    candidate = make_sample_paper(title='Fresh')
    executor = _delivery_executor(config, [candidate])
    executor._search_mode = 5 if failure == 'query' else 2
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda corpus: corpus
    executor.retrievers = {'pubmed': SimpleNamespace(retrieve_papers=lambda: [candidate])}
    monkeypatch.setattr('zotero_arxiv_daily.executor.load_ready', lambda *args: [cached])
    monkeypatch.setattr('zotero_arxiv_daily.executor.focus_evidence', lambda *args: None)
    monkeypatch.setattr('zotero_arxiv_daily.executor.render_email', lambda papers, **kwargs: 'rendered')
    def fail(*args, **kwargs):
        raise RuntimeError('private details must not be logged')
    target = 'build_zotero_queries' if failure == 'query' else 'enrich_journal_metrics'
    monkeypatch.setattr(f'zotero_arxiv_daily.executor.{target}', fail)
    executor.generate_email_summary = fail
    monkeypatch.chdir(tmp_path)
    assert executor.run(send_mail=False) == 1


def test_approved_cognitive_stock_survives_library_outage(config, monkeypatch, tmp_path):
    from tests.test_curation import paper, thought
    from tests.test_executor import _delivery_executor
    config.recommendations.card_cache_enabled = True
    config.recommendations.enabled = True
    config.recommendations.exclude_sent = False
    config.recommendations.max_papers = 2
    config.recommendations.focus_mode = 'balanced'
    config.executor.max_paper_num = 2
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    for number in (1, 2):
        p = paper(number)
        p._approval_basis, p.thought_card = 'cognitive_transfer', thought()
        assert save_ready(p, config, [SimpleNamespace(title='library', abstract='material')])
    executor = _delivery_executor(config, [])
    executor._search_mode = 5
    def unavailable(*args, **kwargs):
        raise RuntimeError('library temporarily unavailable')
    executor.fetch_zotero_corpus = unavailable
    executor.retrievers = {'pubmed': SimpleNamespace(retrieve_papers=lambda: pytest.fail('stock must avoid retrieval'))}
    executor.generate_email_summary = lambda *args: pytest.fail('stock must avoid LLM')
    monkeypatch.chdir(tmp_path)
    assert executor.run(send_mail=False) == 2
