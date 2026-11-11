import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from zotero_arxiv_daily import curation
from zotero_arxiv_daily.card_store import load_ready, save_ready
from zotero_arxiv_daily.protocol import Paper


@pytest.fixture(autouse=True)
def prepared_history(tmp_path):
    from zotero_arxiv_daily.paper_identity import record_sent
    record_sent(tmp_path / 'history.json', [], datetime.now(timezone.utc).date())


def thought():
    return {'scientific_contribution': 'high', 'thought_inspiration': 'medium', 'resource_feasibility': 'low',
            'original_question': '【摘要】作者研究疾病转变吗？', 'reframed_question': '【推断】群体状态是否改变？',
            'key_quote': 'A useful literal quote', 'transfer_action': '【推断】将状态模型用于人群。',
            'testable_question': '【假设】状态变化与结局有关。', 'data_requirements': '【待核验】需要随访数据。',
            'conditions': '【待核验】需要可靠时间顺序。', 'missing_evidence': '【待核验】外部验证未提供。'}


def paper(number=1):
    sentence = '这项研究基于所提供的摘要材料进行审慎分析并报告相关结果。' * 2
    card = {'essence': [sentence, sentence],
            'design': dict.fromkeys(('objects_data', 'task_definitions', 'implementation', 'comparison_validation_scope'), sentence),
            'evidence': [{'claim': '结果', 'comparison': '比较', 'result': '观察', 'locator': '摘要段落1'}],
            'innovation': [sentence, sentence], 'criticism': [sentence, sentence, sentence], 'self_test': '怎样验证？'}
    return Paper(source='pubmed', title=f'Public {number}', authors=['Author'], abstract='A useful literal quote in this abstract.',
                 url=f'https://pubmed.ncbi.nlm.nih.gov/{number}/', journal=f'Journal {number}', discovery_channels=['frontier'],
                 publication_date=datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
                 journal_metrics={'jif': 12}, abstract_zh='中文摘要', reading_card=card,
                 material_scope='仅据摘要（正文与补充材料未提供）', reader_status='success')


def test_thought_evidence_and_resource():
    item = thought()
    assert curation.validate_thought_card(item, paper().abstract)
    item['scientific_contribution'] = 'uncertain'
    assert not curation.validate_thought_card(item, paper().abstract)
    item = thought()
    item['key_quote'] = 'Invented literal quote'
    assert not curation.validate_thought_card(item, paper().abstract)
    item = thought()
    item['original_question'] = 'unknown'
    assert not curation.validate_thought_card(item, paper().abstract)


def test_cache_revalidates_thought_and_keeps_public_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True}})
    corpus = [SimpleNamespace(title='private', abstract='library')]
    p = paper()
    p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
    p.discovery_channels = ['cross_disciplinary']
    p.source_metadata = [{'provider': 'Europe PMC', 'channel': 'cross_disciplinary', 'pmid': '1', 'private': 'secret'}]
    assert save_ready(p, config, corpus)
    stored = next(tmp_path.glob('*.json'))
    assert 'secret' not in stored.read_text(encoding='utf-8')
    loaded = load_ready(config, corpus)
    assert loaded[0].thought_card == thought()
    assert loaded[0].source_metadata[0]['provider'] == 'Europe PMC'
    data = json.loads(stored.read_text(encoding='utf-8'))
    data['paper']['thought_card']['key_quote'] = 'invented quote'
    stored.write_text(json.dumps(data), encoding='utf-8')
    assert load_ready(config, corpus) == []


def test_inventory_failure_preserves_stock_and_rerun_skips_complete(tmp_path, monkeypatch):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 10000}}})
    corpus = [SimpleNamespace(title='unrelated Zotero paper', abstract='library')]
    client = SimpleNamespace(deadline=None)
    executor = SimpleNamespace(openai_client=client, fetch_zotero_corpus=lambda: corpus, filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    p = paper()
    p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
    assert save_ready(p, config, corpus)
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([paper(), paper(2)], [{'status': 'ok'}]))
    def metrics(items, settings):
        assert [item.title for item in items] == ['Public 2']
        return items
    monkeypatch.setattr(curation, 'enrich_journal_metrics', metrics)
    monkeypatch.setattr(curation, 'evaluate_batch', lambda *args: (_ for _ in ()).throw(RuntimeError('endpoint down')))
    result = curation.curate(config)
    assert result['status'] == 'stock_unavailable'
    assert result['stock_before'] == result['stock_after'] == 1
    assert len(load_ready(config, corpus)) == 1


def test_model_batch_does_not_require_zotero_match():
    import time
    item = dict(thought(), id='0')
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'items': [item]})))])
    def create(**kwargs):
        assert 'unrelated Zotero paper' not in json.dumps(kwargs)
        assert kwargs['timeout'] <= 180
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    p = paper()
    assert curation.evaluate_batch([p], client, 'model', time.monotonic() + 60) == [p]
    assert p._approval_basis == 'cognitive_transfer'
    assert p.innovation_assessment['type'] == 'uncertain'


def test_successful_preparation_is_saved_and_not_regenerated(tmp_path, monkeypatch):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 10000}}})
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(deadline=None), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([paper()], [{'status': 'ok'}]))
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings: items)
    calls = []
    OmegaConf.set_struct(config, True)
    def approve(items, *args):
        calls.append(len(items))
        for p in items:
            p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        return items
    monkeypatch.setattr(curation, 'evaluate_batch', approve)
    monkeypatch.setattr(Paper, 'generate_tldr', lambda *args: None)
    assert curation.curate(config)['prepared'] == 1
    assert curation.curate(config)['prepared'] == 0
    assert calls == [1]
    assert len(load_ready(config, corpus)) == 1



def test_empty_discovery_is_not_an_outage_and_two_batch_failures_stop(tmp_path, monkeypatch):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 10000}}})
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(deadline=None), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings: items)
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([], [{'status': 'true_empty'}]))
    assert curation.curate(config)['status'] == 'no_verified_stock'
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([paper(i) for i in range(1, 21)], [{'status': 'success'}]))
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr(curation, 'evaluate_batch', fail)
    assert curation.curate(config)['status'] == 'stock_unavailable'
    assert len(calls) == 2





def test_offline_cognitive_stock_requires_fresh_matching_config_and_valid_evidence(tmp_path, monkeypatch):
    from zotero_arxiv_daily.card_store import load_offline_ready
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'zotero': {'user_id': '123', 'include_path': ['research/**'], 'ignore_path': None, 'api_key': 'secret'},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'min_jif': 10}})
    corpus = [SimpleNamespace(title='private', abstract='library')]
    p = paper()
    p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
    assert save_ready(p, config, corpus)
    path = next(tmp_path.glob('*.json'))
    original = json.loads(path.read_text(encoding='utf-8'))
    assert 'secret' not in path.read_text(encoding='utf-8')
    assert load_offline_ready(config)[0]._cache_basis == 'approved_cognitive_stock_library_unavailable'
    config.zotero.api_key = 'rotated'
    assert len(load_offline_ready(config)) == 1
    config.zotero.user_id = '456'
    assert load_offline_ready(config) == []
    config.zotero.user_id = '123'
    altered = dict(original, saved_at=(datetime.now(timezone.utc) - timedelta(hours=25)).isoformat())
    path.write_text(json.dumps(altered), encoding='utf-8')
    assert load_offline_ready(config) == []
    altered = dict(original)
    altered.pop('configuration_policy')
    path.write_text(json.dumps(altered), encoding='utf-8')
    assert load_offline_ready(config) == []
    altered = dict(original, approval_basis='library')
    path.write_text(json.dumps(altered), encoding='utf-8')
    assert load_offline_ready(config) == []
    altered = json.loads(json.dumps(original))
    altered['paper']['thought_card']['key_quote'] = 'fake quote not in the source'
    path.write_text(json.dumps(altered), encoding='utf-8')
    assert load_offline_ready(config) == []
