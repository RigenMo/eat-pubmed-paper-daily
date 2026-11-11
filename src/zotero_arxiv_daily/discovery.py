"""Public, opt-in discovery experiment. Never receives Zotero library material."""
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, wait
from html.parser import HTMLParser
import math
import os
import re
import requests

from .paper_identity import paper_keys
from .protocol import Paper
from .retriever.pubmed_retriever import _parse_pubmed_xml
from .retriever.public_index import fetch_public_records

URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
# Europe PMC documents CITES:ID_SOURCE as a one-hop citing-publication search:
# https://dev.europepmc.org/docs/EBI_Europe_PMC_Web_Service_Reference.pdf
SEEDS = ('8521067', '21195304')
CHANNELS = ('frontier', 'cross_disciplinary', 'citations', 'counterevidence')
WEIGHTS = (0.25, 0.35, 0.25, 0.15)


def merge_candidates(papers):
    """Merge transitive aliases and retain every public channel/source."""
    groups = []
    for paper in papers:
        keys = paper_keys(paper)
        matched = [group for group in groups if keys & group[1]]
        if not matched:
            groups.append([paper, set(keys)])
            continue
        first = matched[0]
        for other in matched[1:]:
            first[1].update(other[1])
            first[0].discovery_channels = list(dict.fromkeys((first[0].discovery_channels or []) + (other[0].discovery_channels or [])))
            first[0].source_metadata = (first[0].source_metadata or []) + (other[0].source_metadata or [])
            groups.remove(other)
        first[1].update(keys)
        if first[0].source != 'pubmed' and paper.source == 'pubmed':
            paper.discovery_channels = list(dict.fromkeys((first[0].discovery_channels or []) + (paper.discovery_channels or [])))
            paper.source_metadata = (first[0].source_metadata or []) + (paper.source_metadata or [])
            first[0] = paper
            continue
        first[0].discovery_channels = list(dict.fromkeys((first[0].discovery_channels or []) + (paper.discovery_channels or [])))
        first[0].source_metadata = (first[0].source_metadata or []) + (paper.source_metadata or [])
    for paper, keys in groups:
        paper._aliases = sorted(keys)
    return [paper for paper, _ in groups]


def sample_candidates(papers, limit=20):
    """Use weights only for initial quotas, then round-robin available channels."""
    if not 1 <= limit <= 20:
        raise ValueError('limit must be between 1 and 20')
    quotas = [math.floor(limit * weight) for weight in WEIGHTS]
    for index in sorted(range(4), key=lambda i: -(limit * WEIGHTS[i] - quotas[i]))[:limit - sum(quotas)]:
        quotas[index] += 1
    pools = [[paper for paper in papers if channel in (paper.discovery_channels or [])] for channel in CHANNELS]
    selected = []
    selected_ids = set()
    def take(index):
        while pools[index]:
            paper = pools[index].pop(0)
            if id(paper) not in selected_ids:
                selected.append(paper)
                selected_ids.add(id(paper))
                return True
        return False
    for index, quota in enumerate(quotas):
        for _ in range(quota):
            if not take(index):
                break
    while len(selected) < limit:
        progress = False
        for index in range(4):
            if len(selected) < limit and take(index):
                progress = True
        if not progress:
            break
    return selected


def _europe_channel(channel, query, size, today, days, get):
    response = get(URL, params={'query': query + ' sort_date:y', 'format': 'json', 'resultType': 'core', 'pageSize': size}, timeout=8)
    response.raise_for_status()
    items = response.json()['resultList']['result']
    if not isinstance(items, list):
        raise ValueError('invalid results')
    papers = []
    for item in items[:size]:
        if not isinstance(item, dict) or item.get('source') != 'MED':
            continue
        pmid = str(item.get('pmid') or item.get('id') or '')
        first = item.get('firstPublicationDate')
        if not pmid.isdigit() or first not in (item.get('electronicPublicationDate'), item.get('printPublicationDate')):
            continue
        paper = _paper(channel, 'Europe PMC', pmid, item.get('title'), item.get('abstractText'), first,
                       ((item.get('journalInfo') or {}).get('journal') or {}).get('title'),
                       [a.get('fullName') for a in (item.get('authorList') or {}).get('author', []) if isinstance(a, dict) and a.get('fullName')],
                       item.get('doi'), 'https://europepmc.org/article/MED/' + pmid, today, days)
        if paper:
            papers.append(paper)
    return papers, len(items[:size])


def _plain(value):
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []
        def handle_data(self, data):
            self.parts.append(data)
    parser = Parser()
    parser.feed(value or '')
    return ' '.join(' '.join(parser.parts).split())


def _paper(channel, provider, pmid, title, abstract, published, journal, authors, doi, url, today, days):
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', published):
            return None
        published_date = date.fromisoformat(published)
        if not today - timedelta(days=days) <= published_date <= today:
            return None
    except (TypeError, ValueError):
        return None
    title, abstract = _plain(title), _plain(abstract)
    if not title or not abstract:
        return None
    source = 'openalex' if provider == 'OpenAlex' else 'pubmed'
    paper = Paper(source=source, title=title, authors=authors or [], abstract=abstract,
                  url=url if source == 'openalex' else 'https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/',
                  journal=journal or None, publication_date=published, discovery_channels=[channel],
                  source_metadata=[{'provider': provider, 'channel': channel, 'pmid': pmid,
                                    'url': url, 'pubmed_verified': source == 'pubmed'}])
    paper.pmid = pmid
    paper.doi = doi or ''
    return paper


def _pubmed_channel(channel, query, size, today, days, get):
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    if channel == 'citations':
        response = get(base + 'elink.fcgi', params={'dbfrom': 'pubmed', 'db': 'pubmed', 'linkname': 'pubmed_pubmed_citedin', 'id': ','.join(SEEDS), 'retmode': 'json'}, timeout=7)
        response.raise_for_status()
        ids = [str(i) for group in response.json()['linksets'] for links in group.get('linksetdbs', []) if links.get('linkname') == 'pubmed_pubmed_citedin' for i in links.get('links', []) if str(i).isdigit()]
        ids = list(dict.fromkeys(ids))[:size]
    else:
        start = (today - timedelta(days=days)).isoformat()
        term = query + f' AND ("{start}"[Date - Publication] : "{today.isoformat()}"[Date - Publication])'
        response = get(base + 'esearch.fcgi', params={'db': 'pubmed', 'term': term, 'retmode': 'json', 'retmax': size, 'sort': 'pub date'}, timeout=7)
        response.raise_for_status()
        result = response.json()['esearchresult']
        ids = result['idlist']
        if not isinstance(ids, list) or not str(result['count']).isdigit() or (int(result['count']) and not ids):
            raise ValueError('invalid search')
        ids = [str(i) for i in ids[:size] if str(i).isdigit()]
    if not ids:
        return [], 0
    response = get(base + 'efetch.fcgi', params={'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'}, timeout=7)
    response.raise_for_status()
    records = _parse_pubmed_xml(response.text)
    if not records:
        raise ValueError('invalid fetch')
    papers = []
    for item in records:
        pmid = str(item.get('PMID') or '')
        if pmid not in ids:
            continue
        abstract = item.get('AB') or ''
        if isinstance(abstract, list):
            abstract = ' '.join(abstract)
        doi = item.get('LID') or ''
        if isinstance(doi, list):
            doi = doi[0] if doi else ''
        paper = _paper(channel, 'PubMed', pmid, item.get('TI'), abstract, item.get('publication_date'), item.get('JT'), item.get('AU'), doi.replace(' [doi]', ''), 'https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/', today, days)
        if paper:
            papers.append(paper)
    return papers, len(records)


def _openalex_channel(channel, size, today, days, get, key):
    url = 'https://api.openalex.org/works'
    headers = {'Authorization': 'Bearer ' + key} if key else {}
    filters = f'from_publication_date:{(today - timedelta(days=days)).isoformat()},to_publication_date:{today.isoformat()}'
    if channel == 'citations':
        response = get(url, params={'filter': 'pmid:' + SEEDS[0], 'per-page': 1}, headers=headers, timeout=6)
        response.raise_for_status()
        seeds = response.json()['results']
        if not isinstance(seeds, list):
            raise ValueError('invalid seed')
        if not seeds:
            return [], 0
        seed = seeds[0]['id'].rsplit('/', 1)[-1]
        if not re.fullmatch(r'W\d+', seed):
            raise ValueError('invalid seed')
        filters += ',cites:' + seed
    params = {'filter': filters, 'per-page': size, 'sort': 'publication_date:desc'}
    if channel != 'citations':
        params['search'] = 'measurement error causal identification dynamic systems latent state'
    response = get(url, params=params, headers=headers, timeout=6)
    response.raise_for_status()
    items = response.json()['results']
    if not isinstance(items, list):
        raise ValueError('invalid results')
    papers = []
    for item in items[:size]:
        words = sorted((pos, word) for word, positions in (item.get('abstract_inverted_index') or {}).items() if isinstance(positions, list) for pos in positions if isinstance(pos, int) and 0 <= pos < 100000)
        journal = ((item.get('primary_location') or {}).get('source') or {}).get('display_name')
        authors = [a['author']['display_name'] for a in item.get('authorships', []) if isinstance(a, dict) and isinstance(a.get('author'), dict) and a['author'].get('display_name')]
        doi = (item.get('doi') or '').removeprefix('https://doi.org/')
        pmid = str((item.get('ids') or {}).get('pmid') or '').rstrip('/').rsplit('/', 1)[-1]
        if not pmid.isdigit():
            pmid = ''
        paper = _paper(channel, 'OpenAlex', pmid, item.get('title'), ' '.join(word for _, word in words), item.get('publication_date'), journal, authors, doi, item.get('id') or '', today, days)
        if paper:
            papers.append(paper)
    return papers, len(items[:size])


def discover(today=None, days=90, per_channel=30, get=requests.get):
    """Retrieve a bounded public candidate pool. Errors are reported per channel."""
    today = today or date.today()
    if days < 1 or not 1 <= per_channel <= 100:
        raise ValueError('invalid discovery bounds')
    base = f'SRC:MED AND FIRST_PDATE:[{(today - timedelta(days=days)).isoformat()} TO {today.isoformat()}]'
    queries = {
        'frontier': '("UK Biobank" OR multimorbidity OR "public health" OR cardiovascular OR ageing)',
        'cross_disciplinary': '("measurement error" OR "causal identification" OR "state transition" OR "dynamic systems" OR "latent state")',
        'citations': '(' + ' OR '.join(f'CITES:{seed}_MED' for seed in SEEDS) + ')',
        'counterevidence': '("negative results" OR "null findings" OR "failed replication" OR "effect reversal" OR "boundary conditions")',
    }
    key = os.environ.get('OPENALEX_API_KEY', '').strip()

    def run(channel):
        query = queries[channel]
        size = min(per_channel, 10) if channel == 'citations' else per_channel
        sources = [('Europe PMC', lambda: _europe_channel(channel, base + ' AND ' + query, size, today, days, get)),
                   ('PubMed', lambda: _pubmed_channel(channel, query, size, today, days, get))]
        if channel == 'cross_disciplinary' or (key and channel == 'citations'):
            sources.append(('OpenAlex', lambda: _openalex_channel(channel, size, today, days, get, key)))
        papers, providers = [], []
        for provider, fetch in sources:
            try:
                found, returned = fetch()
                papers.extend(found)
                providers.append({'provider': provider, 'status': 'success' if returned else 'true_empty', 'returned': returned, 'accepted': len(found), 'metadata_filtered': returned - len(found)})
            except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError, IndexError):
                providers.append({'provider': provider, 'status': 'unavailable', 'returned': None})
        if not key and channel == 'citations':
            providers.append({'provider': 'OpenAlex', 'status': 'key_required_for_extra_requests', 'returned': None})
        available = [p for p in providers if p['status'] in ('success', 'true_empty')]
        status = 'success' if any(p['status'] == 'success' for p in available) else ('unavailable' if any(p['status'] == 'unavailable' for p in providers) else 'true_empty')
        if channel == 'citations':
            papers = merge_candidates(papers)[:10]
        return papers, {'channel': channel, 'status': status, 'providers': providers,
                        'returned': sum(p['returned'] for p in available) if available else None, 'accepted': len(papers)}

    executor = ThreadPoolExecutor(max_workers=4)
    futures = {channel: executor.submit(run, channel) for channel in CHANNELS}
    done, _ = wait(futures.values(), timeout=34)
    papers, audit = [], []
    for channel, future in futures.items():
        if future in done:
            try:
                found, item = future.result()
                papers.extend(found)
                audit.append(item)
                continue
            except Exception:
                pass
        future.cancel()
        audit.append({'channel': channel, 'status': 'unavailable', 'returned': None, 'accepted': 0, 'providers': []})
    executor.shutdown(wait=False, cancel_futures=True)
    # OpenAlex identifiers are leads, not PubMed verification. Verify only public PMIDs.
    pmids = list(dict.fromkeys(p.pmid for p in papers if p.source == 'openalex' and p.pmid))[:100]
    if pmids:
        for record in fetch_public_records(pmids, max_requests=1):
            pmid = str(record.get('PMID') or '')
            for original in [p for p in papers if p.source == 'openalex' and p.pmid == pmid]:
                doi = str(record.get('LID') or '').replace(' [doi]', '')
                verified = _paper(original.discovery_channels[0], 'Europe PMC', pmid, record.get('TI'), record.get('AB'), record.get('publication_date'), record.get('JT'), record.get('AU'), doi, 'https://europepmc.org/article/MED/' + pmid, today, days)
                if verified:
                    papers.append(verified)
    return merge_candidates(papers), audit

