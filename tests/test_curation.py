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


@pytest.mark.parametrize('field', curation.THOUGHT_FIELDS[:3])
@pytest.mark.parametrize('value', [['high'], {'level': 'high'}, None])
def test_nested_review_ratings_are_rejected_without_type_error(field, value):
    item = dict(thought(), **{field: value})
    assert curation.validate_thought_card(item, paper().abstract) is False


@pytest.mark.parametrize('response_kind', ['fenced', 'nested_rating', 'truncated', 'empty', 'non_object', 'invalid_items'])
def test_reviewer_uses_strict_shared_parser_and_rejects_malformed_cards(response_kind):
    import time
    item = dict(thought(), id='0', thought_inspiration='high')
    content = json.dumps({'items': [item]})
    finish_reason = 'stop'
    if response_kind == 'fenced':
        content = '```json\n' + content + '\n```'
    elif response_kind == 'nested_rating':
        content = json.dumps({'items': [dict(item, scientific_contribution={'level': 'high'})]})
    elif response_kind == 'truncated':
        finish_reason = 'length'
    elif response_kind == 'empty':
        content = None
    elif response_kind == 'non_object':
        content = '[]'
    elif response_kind == 'invalid_items':
        content = '{"items": {}}'
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))])
    calls = []
    def create(**kwargs):
        calls.append(True)
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    p = paper()
    if response_kind in ('fenced', 'nested_rating'):
        result = curation.evaluate_batch([p], client, 'model', time.monotonic() + 60)
        assert result == ([p] if response_kind == 'fenced' else [])
    else:
        with pytest.raises(ValueError):
            curation.evaluate_batch([p], client, 'model', time.monotonic() + 60)
        assert p.thought_card is None
    assert calls == [True]


def test_cache_revalidates_thought_and_keeps_public_provenance(tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cards'
    monkeypatch.setenv('CARD_STORE_DIR', str(cache_dir))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True}})
    corpus = [SimpleNamespace(title='private', abstract='library')]
    p = paper()
    p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
    p.discovery_channels = ['cross_disciplinary']
    p.source_metadata = [{'provider': 'Europe PMC', 'channel': 'cross_disciplinary', 'pmid': '1', 'private': 'secret'}]
    assert save_ready(p, config, corpus)
    stored = next(cache_dir.glob('*.json'))
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
    def metrics(items, settings, **kwargs):
        assert [item.title for item in items] == ['Public 2']
        return items
    monkeypatch.setattr(curation, 'enrich_journal_metrics', metrics)
    monkeypatch.setattr(curation, 'evaluate_batch', lambda *args: (_ for _ in ()).throw(RuntimeError('endpoint down')))
    result = curation.curate(config)
    assert result['status'] == 'stock_unavailable'
    assert result['stock_before'] == result['stock_after'] == 1
    assert len(load_ready(config, corpus)) == 1


@pytest.mark.parametrize('contribution,inspiration', [('high', 'high'), ('medium', 'high'), ('high', 'medium')])
def test_model_batch_requires_strong_contribution_and_transfer_without_zotero_match(contribution, inspiration):
    import time
    item = dict(thought(), id='0', scientific_contribution=contribution, thought_inspiration=inspiration)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'items': [item]})))])
    def create(**kwargs):
        assert 'unrelated Zotero paper' not in json.dumps(kwargs)
        assert kwargs['timeout'] <= 240
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    p = paper()
    result = curation.evaluate_batch([p], client, 'model', time.monotonic() + 60)
    if contribution == inspiration == 'high':
        assert result == [p]
        assert p._approval_basis == 'cognitive_transfer'
        assert p.innovation_assessment['type'] == 'uncertain'
    else:
        assert result == []
        assert p.thought_card is None


@pytest.mark.parametrize('remaining,expected_timeout', [(45, 45), (1000, 720)])
def test_background_review_request_budget_stays_within_whole_deadline(monkeypatch, remaining, expected_timeout):
    monkeypatch.setattr(curation.time, 'monotonic', lambda: 100)
    item = dict(thought(), id='0', thought_inspiration='high')
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'items': [item]})))])
    observed = []
    def create(**kwargs):
        observed.append((kwargs['timeout'], kwargs['max_tokens']))
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert len(curation.evaluate_batch([paper()], client, 'model', 100 + remaining)) == 1
    assert observed == [(expected_timeout, 65536)]


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
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings, **kwargs: items)
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


@pytest.mark.parametrize('configured_tokens,expected_tokens', [(10000, 10000), (65536, 65536),
                                                            (100000, 65536), (None, 65536)])
def test_background_reviews_one_paper_at_a_time_and_preserves_reader_budget(tmp_path, monkeypatch, configured_tokens, expected_tokens):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model'}}})
    if configured_tokens is not None:
        config.llm.generation_kwargs.max_tokens = configured_tokens
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(deadline=None), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([paper(i) for i in range(1, 4)], [{'status': 'ok'}]))
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings, **kwargs: items)
    reviews, readers = [], []
    def approve(items, *args):
        reviews.append(len(items))
        for p in items:
            p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        return items
    def read(paper, client, settings):
        readers.append((settings.generation_kwargs.timeout, settings.generation_kwargs.max_tokens))
    monkeypatch.setattr(curation, 'evaluate_batch', approve)
    monkeypatch.setattr(Paper, 'generate_tldr', read)
    assert curation.curate(config, max_new=3)['prepared'] == 3
    assert reviews == [1, 1, 1] and readers == [(720, expected_tokens)] * 3
    assert config.llm.generation_kwargs.get('timeout') is None
    assert config.llm.generation_kwargs.get('max_tokens') == configured_tokens



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
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings, **kwargs: items)
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
    cache_dir = tmp_path / 'cards'
    monkeypatch.setenv('CARD_STORE_DIR', str(cache_dir))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'zotero': {'user_id': '123', 'include_path': ['research/**'], 'ignore_path': None, 'api_key': 'secret'},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'min_jif': 10}})
    corpus = [SimpleNamespace(title='private', abstract='library')]
    p = paper()
    p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
    assert save_ready(p, config, corpus)
    path = next(cache_dir.glob('*.json'))
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


def test_curation_can_yield_before_starting_any_generation():
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True}})
    result = curation.curate(config, should_yield=lambda: True)
    assert result['status'] == 'paused_for_sender'
    assert result['stock_count_verified'] is False


def test_count_only_revalidates_seen_duplicates_and_journal_cap_without_generation(tmp_path, monkeypatch):
    from zotero_arxiv_daily.paper_identity import record_sent
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True}})
    corpus = [paper(21)]
    executor = SimpleNamespace(openai_client=SimpleNamespace(), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items, zotero_read_titles={'public 22'})
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    record_sent(tmp_path / 'history.json', [paper(20)], datetime.now(timezone.utc).date())
    cached = [paper(i) for i in range(1, 25)]
    for p in cached:
        p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        if int(p.title.split()[-1]) <= 4:
            p.journal = ' Shared   Journal '
        if p.title == 'Public 24':
            p.title = 'Public 23'  # Different identifier, same normalized title.
        assert save_ready(p, config, corpus)
    unexpected_calls = []
    def unexpected(*args, **kwargs):
        unexpected_calls.append(True)
        raise AssertionError('count-only must not discover, review, or generate')
    monkeypatch.setattr(curation, 'discover', unexpected)
    monkeypatch.setattr(curation, 'evaluate_batch', unexpected)
    monkeypatch.setattr(curation, 'enrich_journal_metrics', unexpected)
    monkeypatch.setattr(Paper, 'generate_tldr', unexpected)
    result = curation.curate(config, count_only=True)
    assert result['stock_before'] == result['stock_after'] == 19
    assert result['stock_count_verified'] is True and result['target_met'] is False
    assert result['status'] == 'stock_partial' and unexpected_calls == []
    assert result['prepared'] == result['discovered'] == 0
    assert max(result['journal_distribution'].values()) == 3
    assert sum(result['journal_distribution'].values()) == 19
    assert len(load_ready(config, corpus)) == 24  # Validation preserves completed cards.


def test_count_only_accepts_twenty_independent_qualified_cards(tmp_path, monkeypatch):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True}})
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    for i in range(1, 21):
        p = paper(i)
        p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        assert save_ready(p, config, corpus)
    result = curation.curate(config, count_only=True)
    assert result['stock_after'] == 20 and result['target_met'] is True
    assert result['stock_count_verified'] is True and result['prepared'] == 0


def test_deferred_candidate_uses_other_papers_instead_of_repeating_failure(tmp_path, monkeypatch):
    from zotero_arxiv_daily.card_store import policy_fingerprint
    from zotero_arxiv_daily.paper_identity import paper_keys
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    control = tmp_path / 'control.json'
    monkeypatch.setenv('INVENTORY_CONTROL_FILE', str(control))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 10000}}})
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(deadline=None), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    first, second = paper(1), paper(2)
    control.write_text(json.dumps({'policy': policy_fingerprint(config, corpus),
        'deferred': {key: (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat() for key in paper_keys(first)}}))
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([first, second], [{'status': 'success'}]))
    observed = []
    def metrics(items, settings, **kwargs):
        observed.extend(p.url for p in items)
        return []
    monkeypatch.setattr(curation, 'enrich_journal_metrics', metrics)
    curation.curate(config)
    assert observed == [second.url]


@pytest.mark.parametrize('failure_stage', ['review', 'reader', 'save'])
def test_failed_identity_stays_deferred_through_round_and_max_backoff(tmp_path, monkeypatch, failure_stage):
    from zotero_arxiv_daily.paper_identity import paper_keys
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    control = tmp_path / 'control.json'
    monkeypatch.setenv('INVENTORY_CONTROL_FILE', str(control))
    clock = [datetime.now(timezone.utc)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz else clock[0].replace(tzinfo=None)
    monkeypatch.setattr(curation, 'datetime', Clock)
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 10000}}})
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: ([paper(1)], [{'status': 'ok'}]))
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings, **kwargs: items)
    reviews, readers, saves = [], [], []
    def approve(items, *args):
        reviews.extend(p.url for p in items)
        if failure_stage == 'review':
            raise ValueError('invalid review')
        for p in items:
            p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        return items
    def read(p, *args):
        readers.append(p.url)
        if failure_stage == 'reader':
            raise ValueError('invalid card')
    def save(p, *args):
        saves.append(p.url)
        return False
    monkeypatch.setattr(curation, 'evaluate_batch', approve)
    monkeypatch.setattr(Paper, 'generate_tldr', read)
    monkeypatch.setattr(curation, 'save_ready', save)
    result = curation.curate(config)
    assert result['prepared'] == 0
    for key in paper_keys(paper(1)):
        assert datetime.fromisoformat(result['deferred'][key]) == clock[0] + timedelta(minutes=180)
    control.write_text(json.dumps({'policy': result['policy'], 'deferred': result['deferred']}), encoding='utf-8')
    # A whole 45-minute run followed by the largest 120-minute retry backoff.
    clock[0] += timedelta(minutes=166)
    before = (len(reviews), len(readers), len(saves))
    second = curation.curate(config)
    assert second['unseen_candidates'] == 0 and second['prepared'] == 0
    assert (len(reviews), len(readers), len(saves)) == before


def test_journal_reaching_cap_is_deferred_before_next_review_call(tmp_path, monkeypatch):
    from zotero_arxiv_daily.paper_identity import paper_keys
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 10000}}})
    corpus = [SimpleNamespace(title='unrelated', abstract='library')]
    executor = SimpleNamespace(openai_client=SimpleNamespace(), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda config: executor)
    for i in (1, 2):
        p = paper(i)
        p.journal = 'Shared Journal'
        p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        assert save_ready(p, config, corpus)
    candidates = [paper(i) for i in (3, 4, 5)]
    candidates[0].journal = candidates[1].journal = ' shared  JOURNAL '
    monkeypatch.setattr(curation, 'discover', lambda **kwargs: (candidates, [{'status': 'ok'}]))
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda items, settings, **kwargs: items)
    reviewed = []
    def approve(items, *args):
        reviewed.extend(p.url for p in items)
        for p in items:
            p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
        return items
    monkeypatch.setattr(curation, 'evaluate_batch', approve)
    monkeypatch.setattr(Paper, 'generate_tldr', lambda *args: None)
    before = datetime.now(timezone.utc)
    result = curation.curate(config)
    assert reviewed == [candidates[0].url, candidates[2].url]
    assert result['prepared'] == 2 and result['journal_distribution']['shared journal'] == 3
    for key in paper_keys(candidates[1]):
        deferred_until = datetime.fromisoformat(result['deferred'][key])
        assert before + timedelta(minutes=120) <= deferred_until <= datetime.now(timezone.utc) + timedelta(minutes=120)
