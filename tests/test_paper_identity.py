import json
import pytest
from zotero_arxiv_daily.paper_identity import filter_unseen, load_history, paper_keys, record_sent


def test_identity_and_bridge():
    pmid = {'url': 'https://pubmed.ncbi.nlm.nih.gov/123/?utm_source=a'}
    both = {'url': 'https://www.ncbi.nlm.nih.gov/pubmed/123/', 'doi': '10.1000/ABC%2Fxyz'}
    doi = {'url': 'https://doi.org/10.1000/abc%2Fxyz'}
    assert paper_keys(pmid) & paper_keys({'pmid': '123'})
    assert paper_keys(pmid) & paper_keys(both)
    assert paper_keys(both) & paper_keys(doi)
    result = filter_unseen([pmid, doi, both], set())
    assert len(result) == 1
    assert paper_keys(result[0]) >= paper_keys(doi)
    assert not filter_unseen([pmid, doi, both], paper_keys(doi))


def test_conservative_titles():
    a = {'title': 'A study of hearts', 'authors': ['Smith'], 'publication_date': '2026'}
    b = {'title': 'A study of heart', 'authors': ['Smith'], 'publication_date': '2026'}
    assert len(filter_unseen([a, b], set())) == 2


def test_history(tmp_path):
    path = tmp_path / 'history.json'
    with pytest.raises(FileNotFoundError):
        load_history(path, required=True)
    paper = {'url': 'https://pubmed.ncbi.nlm.nih.gov/123/'}
    record_sent(path, [paper], '2026-11-08')
    assert not filter_unseen([paper], set(load_history(path)['entries']))
    path.write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError):
        load_history(path)
