import json
from types import SimpleNamespace

from zotero_arxiv_daily.construct_email import get_block_html
from zotero_arxiv_daily.reading_extensions import extend, synthesize_week


def paper(n):
    return SimpleNamespace(source='pubmed', reader_status='success', reading_card={'x': 1},
        title='Title', abstract=f'This is the literal evidence for paper {n}.',
        url=f'https://pubmed.ncbi.nlm.nih.gov/{n}/')


def client(payload):
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class Response:
    status_code = 200
    def __init__(self, data): self.data = data
    def raise_for_status(self): pass
    def json(self): return self.data
    @property
    def content(self): return self.data


class Session:
    def __init__(self, oa='Y', license='https://creativecommons.org/licenses/by/4.0/'):
        self.oa, self.license, self.count = oa, license, 0
    def get(self, url, **kwargs):
        self.count += 1
        if url.endswith('/search'):
            return Response({'resultList': {'result': [{'pmid': '1', 'pmcid': 'PMC1', 'isOpenAccess': self.oa}]}})
        return Response(f'''<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
          <article-id pub-id-type="pmc">1</article-id><permissions><license xlink:href="{self.license}"/></permissions>
          </article-meta></front><body><sec><title>Methods</title><p>Long narrative evidence here.</p>
          <fig><caption><p>Secret figure caption.</p></caption></fig></sec></body></article>'''.encode())


def test_oa_and_scope_boundaries():
    c = client({'report': '根据方法节【事实】', 'evidence': [{'quote': 'Long narrative evidence'}]})
    denied = Session(oa='N')
    assert extend([paper(1)], c, {'model': 'x'}, denied)[0]['status'] == 'unavailable'
    assert denied.count == 1
    assert extend([paper(1)], c, {'model': 'x'}, Session(license='https://example.org/'))[0]['status'] == 'unavailable'
    result = extend([paper(1)], c, {'model': 'x'}, Session())[0]
    assert result['status'] == 'success'
    assert 'Results' in result['missing_or_unrecognized_sections']
    assert 'Discussion' in result['missing_or_unrecognized_sections']
    assert '未读取' in result['supplement']
    assert 'Secret figure' not in json.dumps(result)


def test_weekly_requires_two_valid_literal_evidence_items():
    item = dict(question='问题', existing_findings='发现', author_claims='主张', hypothesis='假设',
        conditions='条件', missing_evidence='待证', evidence=[
          {'id': '1', 'quote': 'literal evidence for paper 1'}, {'id': '2', 'quote': 'literal evidence for paper 2'}])
    papers = [paper(1), paper(2)]
    result = synthesize_week(papers, client({'questions': [item]}), {'model': 'x'})
    assert result[0]['evidence'][0]['url'] == papers[0].url
    item['evidence'][1]['quote'] = 'fabricated quotation'
    assert synthesize_week(papers, client({'questions': [item]}), {'model': 'x'}) == []
    item['evidence'][1] = {'id': '3', 'quote': 'literal evidence for paper 2'}
    assert synthesize_week(papers, client({'questions': [item]}), {'model': 'x'}) == []
    assert synthesize_week(papers[:1], client({'questions': [item]}), {'model': 'x'}) == []


def test_thought_card_escape_and_trace():
    thought = {'original_question': '<script>alert(1)</script>', 'key_quote': 'literal evidence for paper 1',
               'transfer_action': '<img src=x onerror=alert(1)>', 'resource_feasibility': 'low'}
    html = get_block_html('Title', 'Author', '', '', 'https://pubmed.ncbi.nlm.nih.gov/1/',
                          source='pubmed', original_abstract=paper(1).abstract, thought_card=thought)
    assert '<script>' not in html and '<img' not in html and '&lt;script&gt;' in html
    assert html.index('可迁移思想卡') < html.index('摘要原文')
    assert 'literal evidence for paper 1' in html
    private = get_block_html('Title', 'A', '', '', 'https://example.org', source='zotero',
                             original_abstract=paper(1).abstract, thought_card=thought)
    assert 'literal evidence for paper 1' not in private.split('摘要原文')[0]
