import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit


def _value(paper, name, default=''):
    return paper.get(name, default) if isinstance(paper, dict) else getattr(paper, name, default)


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def paper_keys(paper):
    keys = {key for key in (_value(paper, 'identity_keys', []) or []) if isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key)}
    keys.update(key for key in (_value(paper, '_aliases', []) or []) if isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key))
    stable = set()
    pmid = str(_value(paper, 'pmid', '') or '').strip()
    if re.fullmatch(r'\d+', pmid):
        stable.add(_hash('pmid:' + pmid))
    urls = [str(_value(paper, field, '') or '') for field in ('url', 'pdf_url')]
    doi = str(_value(paper, 'doi', '') or '').strip()
    for url in urls:
        parts = urlsplit(url)
        host = (parts.hostname or '').casefold()
        path = unquote(parts.path)
        match = None
        if host in ('pubmed.ncbi.nlm.nih.gov', 'www.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov'):
            match = re.match(r'^/(?:pubmed/)?(\d+)(?:/|$)', path, re.I)
        if match:
            stable.add(_hash('pmid:' + match.group(1)))
        if host in ('doi.org', 'dx.doi.org', 'www.doi.org'):
            stable.add(_hash('doi:' + path.lstrip('/').strip().casefold()))
    if doi:
        doi = unquote(doi).strip()
        doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi, flags=re.I)
        doi = re.sub(r'^doi:\s*', '', doi, flags=re.I).strip().casefold()
        if doi:
            stable.add(_hash('doi:' + doi))
    if stable:
        return keys | stable
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme in ('http', 'https') and parts.hostname:
            query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.casefold().startswith('utm_') and k.casefold() not in ('fbclid', 'gclid')]
            normalized = urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip('/') or '/', urlencode(sorted(query)), ''))
            keys.add(_hash('url:' + normalized))
            return keys
    title = ' '.join(unicodedata.normalize('NFKC', str(_value(paper, 'title', '') or '')).casefold().split())
    authors = _value(paper, 'authors', []) or []
    author = authors[0] if isinstance(authors, (list, tuple)) and authors else ''
    year = re.search(r'\b\d{4}\b', str(_value(paper, 'publication_date', '') or ''))
    if title and author and year:
        author = ' '.join(unicodedata.normalize('NFKC', str(author)).casefold().split())
        keys.add(_hash(f'title:{title}|author:{author}|year:{year.group()}'))
    return keys


def load_history(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f'Required sent-paper history missing: {path}')
        return {'version': 1, 'entries': {}, 'reconciled_at': datetime.now(timezone.utc).isoformat(), 'retention_days': 180}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('entries'), dict) or not isinstance(data.get('retention_days'), int) or data['retention_days'] <= 0:
        raise ValueError('Invalid sent-paper history schema')
    try:
        datetime.fromisoformat(data['reconciled_at'].replace('Z', '+00:00'))
        for key, value in data['entries'].items():
            if not re.fullmatch(r'[0-9a-f]{64}', key) or not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError('Invalid history entry')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid sent-paper history') from exc
    return data


def filter_unseen(papers, seen_keys):
    groups = []
    for paper in papers:
        keys = paper_keys(paper)
        matches = [group for group in groups if keys & group[1]]
        if not matches:
            groups.append([paper, set(keys)])
        else:
            first = matches[0]
            first[1].update(keys)
            for other in matches[1:]:
                first[1].update(other[1])
                groups.remove(other)
    result = []
    for paper, keys in groups:
        if keys & seen_keys:
            continue
        if isinstance(paper, dict):
            paper['_aliases'] = sorted(keys)
        else:
            setattr(paper, '_aliases', sorted(keys))
        result.append(paper)
    return result


def record_sent(path, papers, day):
    path = Path(path)
    history = load_history(path)
    day = day.isoformat() if isinstance(day, date) else day
    date.fromisoformat(day)
    for paper in papers:
        for key in paper_keys(paper):
            history['entries'][key] = day
    history['reconciled_at'] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as temp:
        temp_path = temp.name
        json.dump(history, temp, indent=2)
    os.replace(temp_path, path)
