"""Inventory acceptance uses actual labels and reviewed transfer, not file counts."""
import copy
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tests.test_curation import paper, thought
from zotero_arxiv_daily import curation
from zotero_arxiv_daily.card_store import FIELDS, load_ready, policy_fingerprint, configuration_fingerprint
from zotero_arxiv_daily.innovation import QUALITY_CONTRACT, quality_ready, valid_transfer
from zotero_arxiv_daily.protocol import Paper


@pytest.mark.parametrize('kind', ['cognitive', 'capability', 'mixed'])
def test_classification_and_transfer_value_are_independent(kind):
    p = paper()
    p.innovation_assessment['type'] = kind
    p.innovation_assessment['domain_relation'] = 'cross_domain'
    p.thought_card = thought()
    assert quality_ready(p)
    p.innovation_assessment['evidence'] = 'invented source quotation'
    assert not quality_ready(p)


@pytest.mark.parametrize('change', ['unsupported', 'version', 'mapping', 'target', 'quote'])
def test_high_ratings_cannot_substitute_for_a_reviewed_bridge(change):
    p = paper()
    item = thought()
    if change == 'unsupported':
        item['transfer_review']['verdict'] = 'uncertain'
    elif change == 'version':
        item['transfer_review']['version'] = 'old'
    elif change == 'mapping':
        item['transfer_review']['mapping'] = '【推断】同词'
    elif change == 'target':
        item['transfer_review']['target'] = 'acoustic_sensor'
    else:
        item['key_quote'] = 'fabricated quote'
    assert not valid_transfer(item, p.abstract)


def prepare_legacy(tmp_path, monkeypatch, basis="cognitive_transfer"):
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path / 'cards'))
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config = OmegaConf.create({
        'recommendations': {'card_cache_enabled': True, 'innovation_enabled': False},
        'journals': {'enabled': True, 'mode': 'easyscholar', 'easyscholar': {'api_key': 'test'}},
        'llm': {'generation_kwargs': {'model': 'model', 'max_tokens': 65536}}})
    corpus = [SimpleNamespace(title='private library title', abstract='private abstract')]
    p = paper()
    p.thought_card, p._approval_basis = thought(), 'cognitive_transfer'
    p.thought_card.pop('transfer_review')
    p.innovation_assessment = {'type': 'uncertain', 'reason': '旧版待判定'}
    # Recreate a structurally valid legacy file without rewriting private inputs.
    from zotero_arxiv_daily.paper_identity import paper_keys
    from datetime import datetime, timezone
    values = {field: getattr(p, field, None) for field in FIELDS}
    config.recommendations.innovation_enabled = True
    payload = {'schema_version': 1, 'policy': policy_fingerprint(config, corpus),
        'configuration_policy': configuration_fingerprint(config), 'saved_at': datetime.now(timezone.utc).isoformat(),
        'approval_basis': basis, 'paper': values, 'identity_keys': sorted(paper_keys(p))}
    path = tmp_path / 'cards' / 'legacy.json'
    path.parent.mkdir()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    executor = SimpleNamespace(openai_client=SimpleNamespace(), fetch_zotero_corpus=lambda: corpus,
                               filter_corpus=lambda items: items)
    monkeypatch.setattr(curation, 'Executor', lambda _: executor)
    monkeypatch.setattr(curation, 'discover', lambda **_: ([], [{'status': 'true_empty'}]))
    monkeypatch.setattr(curation, 'enrich_journal_metrics', lambda *a, **k: [])
    monkeypatch.setattr(Paper, 'generate_tldr', lambda *a: pytest.fail('legacy reader must not run'))
    return config, corpus, p, path


def test_pending_stock_does_not_count_and_metadata_only_recovery_preserves_reader(tmp_path, monkeypatch):
    config, corpus, old, path = prepare_legacy(tmp_path, monkeypatch)
    before = path.read_bytes()
    calls = []
    def review(items, *args):
        calls.append(1)
        for p in items:
            p.thought_card = thought()
            p.innovation_assessment = copy.deepcopy(paper().innovation_assessment)
        return items
    monkeypatch.setattr(curation, 'evaluate_batch', review)
    assert load_ready(config, corpus) == []
    result = curation.curate(config, count_only=True)
    assert result['stock_after'] == 0 and result['pending_review'] == 1
    assert calls == [] and path.read_bytes() == before
    result = curation.curate(config, max_new=1)
    assert result['stock_before'] == 0 and result['stock_after'] == 1
    assert result['prepared'] == 0 and result['quality_revalidated'] == 1
    assert result['classification_counts'] == {'cognitive': 1, 'capability': 0, 'mixed': 0}
    assert result['transfer_required'] == result['transfer_verified'] == 1
    restored = load_ready(config, corpus)
    assert len(restored) == 1
    assert {key: restored[0].reading_card[key] for key in old.reading_card} == old.reading_card
    assert restored[0].abstract_zh == old.abstract_zh
    assert quality_ready(restored[0])


def test_failed_metadata_review_keeps_legacy_file_without_qualifying_it(tmp_path, monkeypatch):
    config, corpus, _, path = prepare_legacy(tmp_path, monkeypatch)
    before = path.read_bytes()
    monkeypatch.setattr(curation, 'evaluate_batch', lambda *a: [])
    result = curation.curate(config)
    assert result['stock_after'] == 0 and result['quality_revalidated'] == 0
    assert path.read_bytes() == before and load_ready(config, corpus) == []
    assert len(load_ready(config, corpus, include_pending=True)) == 1


@pytest.mark.parametrize('field,value', [
    ('quality_contract', 'old'),
    ('classification_counts', {'cognitive': 19, 'capability': 0, 'mixed': 0}),
    ('transfer_verified', 11)])
def test_full_restore_rejects_incomplete_classification_or_transfer_coverage(tmp_path, field, value):
    from tests.test_inventory_priority import module
    summary = {'policy': 'policy', 'sent_ledger_digest': 'ledger', 'quality_contract': QUALITY_CONTRACT,
        'stock_after': 20, 'stock_count_verified': True, 'target_met': True, 'status': 'ok',
        'classification_counts': {'cognitive': 10, 'capability': 8, 'mixed': 2},
        'transfer_required': 12, 'transfer_verified': 12}
    a, b = tmp_path / 'summary.json', tmp_path / 'restored.json'
    a.write_text(json.dumps(summary), encoding='utf-8')
    b.write_text(json.dumps(dict(summary, **{field: value})), encoding='utf-8')
    assert module.finalize_restored_stock(a, b)['durable_stock_verified'] is False


def test_direct_legacy_stock_only_needs_classification_not_an_invented_bridge(tmp_path, monkeypatch):
    config, corpus, old, _ = prepare_legacy(tmp_path, monkeypatch, basis='library')
    def classify(items, *args):
        for p in items:
            p.innovation_assessment = copy.deepcopy(paper().innovation_assessment)
            p.thought_card = None
        return items
    monkeypatch.setattr(curation, 'assess_innovation', classify)
    monkeypatch.setattr(curation, 'evaluate_batch', lambda *a: pytest.fail('direct stock must not invent a transfer'))
    result = curation.curate(config, max_new=1)
    assert result['stock_after'] == result['quality_revalidated'] == 1
    assert result['transfer_required'] == result['transfer_verified'] == 0
    recovered = load_ready(config, corpus)[0]
    assert recovered._approval_basis == 'library' and recovered.thought_card is None
    assert recovered.abstract_zh == old.abstract_zh


def test_quality_contract_change_and_metadata_growth_resume_refill(tmp_path):
    from tests.test_inventory_replenishment import node
    program = '''
import assert from 'node:assert/strict';
const {advance,decide}=await import(process.env.MODULE); const now=Date.now();
const old={stock_after:20,stock_verified:true,ledger_digest:'ledger',
 policy:'same',quality_contract:'old',completed_at:new Date(now).toISOString()};
assert.equal(decide(old,now,'ledger').reason,'quality_review_due');
const summary={quality_contract:'classification-transfer-v1',policy:'same',
 stock_count_verified:true,sent_ledger_digest:'ledger',stock_after:2,prepared:0,
 quality_revalidated:2,status:'stock_partial'};
const current=advance(old,summary,now,'ledger',1);
assert.equal(current.continue_now,true); assert.equal(current.stock_after,2);
const flat=advance(current,summary,now+1000,'ledger',2);
assert.equal(flat.continue_now,false); console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)
