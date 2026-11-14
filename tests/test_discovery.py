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


def test_public_retrieval_uses_four_bounded_channels(monkeypatch):
    import requests
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    calls = []
    def get(url, params, timeout, **kwargs):
        calls.append((url, params, timeout))
        if 'europepmc' not in url:
            raise requests.ConnectionError()
        channel_id = next(i for i, term in enumerate(('UK Biobank', 'measurement error', 'CITES:', 'negative results'), 1) if term in params['query'])
        item = {'source': 'MED', 'id': str(channel_id), 'title': 'Public', 'abstractText': 'A &amp; B.',
                'firstPublicationDate': '2026-11-01', 'electronicPublicationDate': '2026-11-01',
                'journalInfo': {'journal': {'title': 'Example Journal'}}, 'authorList': {'author': [{'fullName': 'First Author'}]}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'resultList': {'result': [item]}})
    result, audit = discover(today=date(2026, 11, 11), per_channel=30, get=get)
    assert len(result) == len(audit) == 4
    assert all(item['status'] == 'success' for item in audit)
    assert all(p.source == 'pubmed' and p.abstract == 'A & B.' for p in result)
    assert all(p.journal == 'Example Journal' for p in result)
    epmc = [params for url, params, _ in calls if 'europepmc' in url]
    assert len(epmc) == 4
    assert next(p for p in epmc if 'CITES:' in p['query'])['pageSize'] == 10
    assert all('FIRST_PDATE:[2026-08-13 TO 2026-11-11]' in p['query'] for p in epmc)
    assert len([1 for url, _, _ in calls if 'openalex.org' in url]) == 1
    assert all(timeout <= 8 for _, _, timeout in calls)


def test_wider_recent_pool_reaches_older_candidates_without_widening_date_or_openalex_limits(monkeypatch):
    import requests
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    calls = []
    items = [{'source': 'MED', 'id': str(i + 1), 'title': f'Public study {i + 1}',
              'abstractText': 'A public source abstract.', 'firstPublicationDate': '2026-10-20' if i >= 100 else '2026-11-01',
              'electronicPublicationDate': '2026-10-20' if i >= 100 else '2026-11-01',
              'journalInfo': {'journal': {'title': 'Example Journal'}}} for i in range(200)]
    def get(url, params, timeout, **kwargs):
        calls.append((url, params, timeout))
        if 'openalex.org' in url:
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': []})
        if 'europepmc' not in url:
            raise requests.ConnectionError()
        found = items if 'UK Biobank' in params['query'] else []
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'resultList': {'result': found}})
    narrow, _ = discover(today=date(2026, 11, 11), per_channel=100, get=get)
    wider, _ = discover(today=date(2026, 11, 11), per_channel=200, get=get)
    assert len(narrow) == 100 and len(wider) == 200
    assert any(p.publication_date == '2026-10-20' for p in wider)
    assert all('FIRST_PDATE:[2026-08-13 TO 2026-11-11]' in p['query'] for u, p, _ in calls if 'europepmc' in u)
    assert all(p['per-page'] <= 100 for u, p, _ in calls if 'openalex.org' in u)
    assert all(timeout <= 8 for _, _, timeout in calls)


def test_parallel_channels_share_ncbi_request_start_limit(monkeypatch):
    import zotero_arxiv_daily.discovery as module
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    now, started = [0.0], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    def get(url, params, timeout, **kwargs):
        if 'eutils.ncbi' in url:
            started.append(now[0])
            data = {'linksets': []} if 'elink' in url else {'esearchresult': {'idlist': [], 'count': '0'}}
        elif 'openalex.org' in url:
            data = {'results': []}
        else:
            data = {'resultList': {'result': []}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    module.discover(today=date(2026, 11, 11), get=get)
    assert len(started) == 4
    assert all(b - a >= 0.349 for a, b in zip(sorted(started), sorted(started)[1:]))


def test_channel_failure_does_not_stop_other_channels(monkeypatch):
    import requests
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    def get(url, params, timeout, **kwargs):
        if 'europepmc' not in url or 'UK Biobank' in params['query']:
            raise requests.ConnectionError()
        channel_id = next(i for i, term in enumerate(('measurement error', 'CITES:', 'negative results'), 2) if term in params['query'])
        item = {'source': 'MED', 'id': str(channel_id), 'title': 'Title', 'abstractText': 'An abstract',
                'firstPublicationDate': '2026-11-01', 'printPublicationDate': '2026-11-01'}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'resultList': {'result': [item]}})
    result, audit = discover(today=date(2026, 11, 11), get=get)
    assert len(result) == 3
    assert audit[0]['status'] == 'unavailable' and audit[0]['returned'] is None
    assert all(item['status'] == 'success' for item in audit[1:])


def test_openalex_only_stays_unverified_and_missing_dates_rejected(monkeypatch):
    import requests
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    def get(url, params, timeout, **kwargs):
        if 'openalex.org' not in url:
            raise requests.Timeout()
        item = {'id': 'https://openalex.org/W123', 'title': 'Public methods', 'publication_date': '2026-11-01',
                'abstract_inverted_index': {'Useful': [0], 'methods': [1]}, 'doi': 'https://doi.org/10.1/example'}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': [item]})
    result, audit = discover(today=date(2026, 11, 11), get=get)
    assert len(result) == 1 and result[0].source == 'openalex'
    assert result[0].abstract == 'Useful methods' and result[0].journal is None
    assert result[0].source_metadata[0]['pubmed_verified'] is False
    assert audit[0]['status'] == 'unavailable' and audit[1]['status'] == 'success'


def test_shortfall_is_round_robin_not_hard_quota():
    pool = [paper(str(i), 'frontier') for i in range(1, 12)] + [paper(str(i), 'cross_disciplinary') for i in range(20, 31)]
    selected = sample_candidates(pool, 10)
    assert sum('frontier' in p.discovery_channels for p in selected) == 5
    assert sum('cross_disciplinary' in p.discovery_channels for p in selected) == 5
    assert selected[0].discovery_channels == ['frontier']
    assert selected[1].discovery_channels == ['cross_disciplinary']


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


def test_real_europe_core_schema_and_explicit_date():
    import json
    from pathlib import Path
    from zotero_arxiv_daily.discovery import _europe_channel
    payload = json.loads((Path(__file__).parents[1] / 'tests/fixtures/europe_core_37512678.json').read_text(encoding='utf-8'))
    get = lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    papers, returned = _europe_channel('frontier', 'public', 1, date(2023, 8, 20), 90, get)
    assert returned == 1 and len(papers) == 1 and papers[0].pmid == '37512678'
    assert papers[0].abstract and papers[0].publication_date
    payload['resultList']['result'][0].pop('electronicPublicationDate', None)
    payload['resultList']['result'][0].pop('printPublicationDate', None)
    papers, returned = _europe_channel('frontier', 'public', 1, date(2023, 8, 20), 90, get)
    assert returned == 1 and papers == []


def test_openalex_pmid_is_upgraded_only_after_real_core_metadata(monkeypatch):
    import requests
    import zotero_arxiv_daily.discovery as module
    monkeypatch.delenv('OPENALEX_API_KEY', raising=False)
    def get(url, params, timeout, **kwargs):
        if 'openalex.org' not in url:
            raise requests.ConnectionError()
        item = {'id': 'https://openalex.org/W123', 'title': 'OpenAlex title', 'publication_date': '2026-11-01',
                'abstract_inverted_index': {'Exploratory': [0]}, 'ids': {'pmid': 'https://pubmed.ncbi.nlm.nih.gov/123/'}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': [item]})
    # A trailing slash is accepted by the OpenAlex external identifier parser.
    item_record = {'PMID': '123', 'TI': 'Verified title', 'AB': 'Verified abstract', 'JT': 'Journal', 'AU': [], 'publication_date': '2026-11-01'}
    monkeypatch.setattr(module, 'fetch_public_records', lambda ids, max_requests: [item_record] if ids == ['123'] else [])
    papers, _ = discover(today=date(2026, 11, 11), get=get)
    assert len(papers) == 1 and papers[0].source == 'pubmed'
    assert papers[0].title == 'Verified title'
    assert {m['provider'] for m in papers[0].source_metadata} == {'OpenAlex', 'Europe PMC'}

