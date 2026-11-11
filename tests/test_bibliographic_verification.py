from datetime import date
from types import SimpleNamespace

from zotero_arxiv_daily.bibliographic_verification import verified_openalex, verify_openalex
from zotero_arxiv_daily.protocol import Paper


def candidate():
    return Paper(source='openalex', title='A scientific comparison', abstract='A concrete abstract.',
                 authors=['Author'], url='https://openalex.org/W1', pdf_url='https://doi.org/10.1234/example',
                 journal='Example Journal', publication_date='2026-11-01', source_metadata=[{'provider': 'OpenAlex'}])


def message():
    return {'DOI': '10.1234/example', 'type': 'journal-article', 'title': ['A scientific comparison'],
            'container-title': ['Example Journal'], 'published': {'date-parts': [[2026, 10, 31]]}}


def response(value):
    return lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, is_redirect=False,
                                                  json=lambda: {'message': value})


def test_independent_doi_title_journal_and_exact_date():
    p = candidate()
    assert not verified_openalex(p)
    assert verify_openalex([p], date(2026, 11, 11), get=response(message())) == [p]
    assert verified_openalex(p) and p.publication_date == '2026-10-31'
    assert p.url == 'https://doi.org/10.1234/example'
    p.publication_date = '2026-11-01'
    assert not verified_openalex(p)


def test_partial_dates_and_mismatched_identity_are_not_approved():
    for change in ({'DOI': '10.1234/other'}, {'title': ['Different paper']},
                   {'container-title': ['Different journal']}, {'published': {'date-parts': [[2026, 10]]}},
                   {'published': {'date-parts': [[2020, 10, 31]]}}):
        value = dict(message(), **change)
        assert verify_openalex([candidate()], date(2026, 11, 11), get=response(value)) == []


def test_core_cache_retains_proof_and_revalidates(tmp_path, monkeypatch):
    from tests.test_curation import paper, thought
    from omegaconf import OmegaConf
    from zotero_arxiv_daily.card_store import load_ready, save_ready
    p = paper()
    p.source, p.pdf_url = 'openalex', 'https://doi.org/10.1234/example'
    p.source_metadata = [{'provider': 'Crossref', 'verification': 'doi_title_journal_date',
                          'doi': '10.1234/example', 'publication_date': p.publication_date, 'journal': p.journal}]
    p._approval_basis, p.thought_card = 'cognitive_transfer', thought()
    config = OmegaConf.create({'recommendations': {'card_cache_enabled': True}})
    monkeypatch.setenv('CARD_STORE_DIR', str(tmp_path))
    assert save_ready(p, config, [])
    assert verified_openalex(load_ready(config, [])[0])
