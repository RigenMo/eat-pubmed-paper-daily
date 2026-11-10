from datetime import date
from types import SimpleNamespace

from zotero_arxiv_daily.discovery import discover, merge_candidates, sample_candidates
from zotero_arxiv_daily.protocol import Paper


def paper(pmid, channel, doi=''):
    p = Paper(source='Europe PMC', title='Title', authors=['Author'], abstract='Some abstract', url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', discovery_channels=[channel], source_metadata=[{'provider': 'Europe PMC', 'channel': channel, 'pmid': pmid}])
    p.doi = doi
    return p


def test_merge_transitive_aliases_and_sources():
    a = paper('1', 'frontier', '10.1/one')
    b = paper('2', 'citations', '10.1/two')
    bridge = paper('1', 'cross_disciplinary', '10.1/two')
    result = merge_candidates([a, b, bridge])
    assert len(result) == 1
    assert set(result[0].discovery_channels) == {'frontier', 'citations', 'cross_disciplinary'}
    assert len(result[0].source_metadata) == 3
    assert len(result[0]._aliases) == 4


def test_sampling_shortfall_fills_without_duplicates():
    pool = [paper(str(i), 'cross_disciplinary') for i in range(1, 15)]
    pool += [paper('20', 'counterevidence')]
    selected = sample_candidates(pool, 10)
    assert len(selected) == 10
    assert len({p.url for p in selected}) == 10
    assert any('counterevidence' in p.discovery_channels for p in selected)


def test_public_retrieval_uses_four_bounded_channels():
    calls = []
    def get(url, params, timeout):
        calls.append(params)
        data = {'resultList': {'result': [{'source': 'MED', 'id': str(len(calls)), 'title': 'Public', 'abstractText': 'Original abstract.', 'firstPublicationDate': '2026-11-01', 'journalInfo': {'journal': {'title': 'Example Journal'}}, 'authorList': {'author': [{'fullName': 'First Author'}]}}]}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    result, audit = discover(today=date(2026, 11, 11), per_channel=30, get=get)
    assert len(result) == len(audit) == 4
    assert result[0].journal == 'Example Journal'
    assert result[0].authors == ['First Author']
    assert result[0].source == 'pubmed'
    assert calls[2]['pageSize'] == 10
    assert 'CITES:8521067_MED' in calls[2]['query']
    assert all('FIRST_PDATE:[2026-08-13 TO 2026-11-11]' in call['query'] for call in calls)
    assert all(set(p.source_metadata[0]) == {'provider', 'channel', 'pmid'} for p in result)


def test_channel_failure_does_not_stop_other_channels():
    import requests
    calls = []
    def get(url, params, timeout):
        calls.append(params)
        if len(calls) == 1:
            raise requests.ConnectionError('unavailable')
        data = {'resultList': {'result': [{'source': 'MED', 'id': str(len(calls)), 'title': 'Title', 'abstractText': 'An abstract', 'firstPublicationDate': '2026-11-01'}]}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    result, audit = discover(today=date(2026, 11, 11), get=get)
    assert len(result) == 3
    assert audit[0]['status'] == 'ConnectionError'
    assert all(item['status'] == 'ok' for item in audit[1:])


def test_shortfall_is_round_robin_not_hard_quota():
    pool = [paper(str(i), 'frontier') for i in range(1, 12)] + [paper(str(i), 'cross_disciplinary') for i in range(20, 31)]
    selected = sample_candidates(pool, 10)
    assert sum('frontier' in p.discovery_channels for p in selected) == 5
    assert sum('cross_disciplinary' in p.discovery_channels for p in selected) == 5


def test_evaluation_rejects_invented_quotes_but_resource_does_not_downrank():
    import importlib.util
    import json
    import time
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('discovery_preflight', Path(__file__).parents[1] / 'scripts' / 'discovery_preflight.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = paper('1', 'frontier')
    item = {'id': '0', 'type': 'cognitive', 'reason': 'reframing', 'evidence': 'Some abstract', 'scientific_contribution': 'high', 'thought_inspiration': 'medium', 'resource_feasibility': 'low', 'key_quote': 'Some abstract'}
    item.update({field: '【推断】待核验' for field in module.FIELDS})
    def client_for(payload):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    result = module.evaluate([p], client_for({'items': [item]}), 'model', time.monotonic() + 5)
    assert result[0]['new_eligible'] is True
    item['key_quote'] = 'invented quotation'
    result = module.evaluate([p], client_for({'items': [item]}), 'model', time.monotonic() + 5)
    assert not result[0]['new_valid'] and not result[0]['new_eligible'] and p.thought_card is None
    result = module.evaluate([p], client_for({'items': 'malformed'}), 'model', time.monotonic() + 5)
    assert result[0]['new'] is None
