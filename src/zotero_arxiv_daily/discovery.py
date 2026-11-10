"""Public, opt-in discovery experiment. Never receives Zotero library material."""
from datetime import date, timedelta
import math
import re
import requests

from .paper_identity import paper_keys
from .protocol import Paper

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
    papers, audit = [], []
    for channel, query in queries.items():
        size = min(per_channel, 10) if channel == 'citations' else per_channel
        try:
            before = len(papers)
            date_filtered = 0
            response = get(URL, params={'query': f'{base} AND {query} sort_date:y', 'format': 'json', 'resultType': 'core', 'pageSize': size}, timeout=20)
            response.raise_for_status()
            results = response.json()['resultList']['result']
            if not isinstance(results, list):
                raise ValueError('invalid results')
            for item in results[:size]:
                pmid = str(item.get('pmid') or item.get('id') or '')
                if not pmid.isdigit() or item.get('source') != 'MED':
                    continue
                title = item.get('title')
                abstract_raw = item.get('abstractText')
                published = item.get('firstPublicationDate')
                if not isinstance(title, str) or not title.strip() or not isinstance(abstract_raw, str) or not abstract_raw.strip():
                    continue
                try:
                    published_date = date.fromisoformat(published)
                except (TypeError, ValueError):
                    date_filtered += 1
                    continue
                if not today - timedelta(days=days) <= published_date <= today:
                    date_filtered += 1
                    continue
                abstract = re.sub(r'<[^>]+>', '', abstract_raw).strip()
                if not abstract:
                    continue
                journal = item.get('journalInfo', {}).get('journal', {}).get('title')
                author_list = item.get('authorList', {}).get('author', [])
                authors = [author.get('fullName') for author in author_list if isinstance(author, dict) and author.get('fullName')]
                paper = Paper(source='pubmed', title=title, authors=authors, abstract=abstract,
                              url='https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/', journal=journal, publication_date=published,
                              discovery_channels=[channel], source_metadata=[{'provider': 'Europe PMC', 'channel': channel, 'pmid': pmid}])
                paper.pmid = pmid
                paper.doi = item.get('doi') or ''
                papers.append(paper)
            audit.append({'channel': channel, 'returned': len(results[:size]), 'accepted': len(papers) - before, 'date_filtered': date_filtered, 'metadata_filtered': len(results[:size]) - (len(papers) - before) - date_filtered, 'status': 'ok'})
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
            audit.append({'channel': channel, 'returned': 0, 'status': type(exc).__name__})
    return merge_candidates(papers), audit
