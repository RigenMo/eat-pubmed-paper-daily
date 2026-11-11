"""Bounded public DOI verification, independent of PubMed/Europe PMC.

Crossref dates can be partial: use original date-parts, never their imputed
timestamps. Public pool requests are sequential (concurrency limit one).
https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication
"""
from datetime import date, timedelta
import re
import time
from urllib.parse import quote, unquote, urlsplit

import requests


def doi_of(paper):
    value = str(getattr(paper, 'doi', '') or '')
    if not value:
        for field in ('pdf_url', 'url'):
            parts = urlsplit(str(getattr(paper, field, '') or ''))
            if parts.hostname in ('doi.org', 'dx.doi.org'):
                value = unquote(parts.path.lstrip('/'))
                break
    value = value.removeprefix('https://doi.org/').strip().casefold()
    return value if re.fullmatch(r'10\.\d{4,9}/\S+', value) else ''


def verified_openalex(paper):
    doi = doi_of(paper)
    return bool(doi) and getattr(paper, 'source', '') == 'openalex' and any(
        item.get('provider') == 'Crossref' and item.get('verification') == 'doi_title_journal_date'
        and item.get('doi') == doi and item.get('publication_date') == paper.publication_date
        and item.get('journal') == paper.journal
        for item in (getattr(paper, 'source_metadata', None) or []) if isinstance(item, dict))


def _normal(value):
    return ''.join(re.findall(r'\w+', value.casefold()))


def _first_date(message):
    dates, partial = [], []
    for field in ('published-online', 'published-print', 'published', 'issued'):
        parts = message.get(field, {}).get('date-parts', [])
        if not parts or not isinstance(parts[0], list):
            continue
        values = parts[0]
        if any(type(v) is not int for v in values):
            return None
        if len(values) == 3:
            dates.append(date(*values))
        elif 1 <= len(values) <= 2:
            partial.append(tuple(values))
    if not dates:
        return None
    first = min(dates)
    exact = (first.year, first.month, first.day)
    if any(p <= exact[:len(p)] for p in partial):
        return None
    return first


def verify_openalex(papers, today, days=90, deadline=None, get=None):
    get = get or requests.get
    deadline = min(deadline or float('inf'), time.monotonic() + 30)
    approved = []
    for paper in list(papers)[:8]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        doi = doi_of(paper)
        if not doi:
            continue
        url = 'https://api.crossref.org/works/' + quote(doi, safe='')
        try:
            response = get(url, timeout=min(6, remaining), allow_redirects=False,
                           headers={'User-Agent': 'EatPaperDaily/1.0 (public metadata verification)'})
            response.raise_for_status()
            if getattr(response, 'is_redirect', False):
                continue
            message = response.json()['message']
            title = (message.get('title') or [''])[0]
            journal = (message.get('container-title') or [''])[0]
            published = _first_date(message)
            if (message.get('type') != 'journal-article' or str(message.get('DOI', '')).casefold() != doi
                    or not _normal(title) or _normal(title) != _normal(paper.title)
                    or not _normal(journal) or _normal(journal) != _normal(paper.journal or '')
                    or published is None or not today - timedelta(days=days) <= published <= today):
                continue
            paper.url = paper.pdf_url = 'https://doi.org/' + doi
            paper.publication_date = published.isoformat()
            paper.source_metadata = (paper.source_metadata or []) + [{
                'provider': 'Crossref', 'verification': 'doi_title_journal_date', 'doi': doi,
                'journal': paper.journal, 'publication_date': paper.publication_date, 'url': url}]
            approved.append(paper)
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError, IndexError):
            continue
        finally:
            time.sleep(0.25)
    return approved
