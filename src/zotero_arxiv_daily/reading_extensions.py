"""Optional no-mail reading extensions; inputs are already approved public cards.

Europe PMC REST fullTextXML: https://europepmc.org/RestfulWebService
PMC OA service retired: https://pmc.ncbi.nlm.nih.gov/tools/oa-service/
"""
import json
import queue
import threading
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlsplit

import requests
from .bibliographic_verification import doi_of, verified_openalex

BASE = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
LICENSE = re.compile(r'https?://(?:www\.)?creativecommons\.org/(?:licenses/(?:by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd)/[1-4]\.0|publicdomain/(?:zero|mark)/1\.0)/?(?=[\s"<>]|$)', re.I)


def _call(client, llm_params, prompt, material, timeout):
    response = client.chat.completions.create(
        model=llm_params['model'], timeout=timeout, max_tokens=6000, response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': prompt},
                  {'role': 'user', 'content': json.dumps(material, ensure_ascii=False)}])
    return json.loads(response.choices[0].message.content)


def _id(paper):
    url = getattr(paper, 'url', '') or ''
    parsed = urlsplit(url)
    if verified_openalex(paper):
        return 'doi:' + doi_of(paper)
    if parsed.scheme != 'https' or parsed.hostname != 'pubmed.ncbi.nlm.nih.gov':
        return ''
    match = re.fullmatch(r'/(\d+)/?', parsed.path)
    return match.group(1) if match else ''


def _eligible(paper):
    return ((getattr(paper, 'source', '') == 'pubmed' or verified_openalex(paper)) and
            getattr(paper, 'reader_status', '') == 'success' and
            bool(getattr(paper, 'reading_card', None)) and
            bool(getattr(paper, 'abstract', None)) and bool(_id(paper)))


def _xml(data):
    if len(data) > 8_000_000 or b'<!ENTITY' in data.upper():
        raise ValueError('unsafe or oversized XML')
    return ET.fromstring(data)


def _remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError('extension deadline')
    return value


def _get(session, url, deadline, **kwargs):
    response = session.get(url, timeout=min(20, _remaining(deadline)), allow_redirects=False, **kwargs)
    response.raise_for_status()
    if 300 <= response.status_code < 400:
        raise ValueError('redirect rejected')
    return response


def _body(xml, pmcid):
    root = _xml(xml)
    ids = [e.text for e in root.findall('.//article-id[@pub-id-type="pmc"]')]
    if not any((i or '').upper().removeprefix('PMC') == pmcid[3:] for i in ids):
        raise ValueError('PMCID mismatch')
    license_urls = []
    for element in root.findall('.//permissions/license'):
        license_urls.extend(str(value) for value in element.attrib.values())
        license_urls.extend(element.itertext())
    matched = LICENSE.search(' '.join(license_urls))
    if not matched:
        raise ValueError('license not explicitly supported')
    body = root.find('.//body')
    if body is None:
        raise ValueError('body unavailable')
    sections = []
    excluded = set()
    for tag in ('fig', 'table-wrap', 'supplementary-material'):
        for node in body.iter(tag):
            excluded.update(node.iter())
    for sec in body.findall('./sec'):
        title = ' '.join(sec.findtext('title', '').split()) or 'Untitled'
        # Only narrative paragraphs: skip figures, tables and supplements.
        paragraphs = []
        for node in sec.iter():
            if node.tag == 'p' and node not in excluded:
                paragraphs.append(' '.join(node.itertext()))
        text = ' '.join(' '.join(paragraphs).split())
        if text:
            sections.append({'heading': title, 'text': text})
    if not sections:
        raise ValueError('body sections unavailable')
    return matched.group(), sections


def _extend(papers, client, llm_params, session=None):
    """Read at most two approved cards, with at most four HTTP requests in 120s.

    Returns a separate report; never changes papers or sends mail. Model input is
    capped and the exact supplied scope is recorded. Supplements are not read.
    """
    session = session or requests.Session()
    deadline = time.monotonic() + 120
    result = []
    for paper in [p for p in papers if _eligible(p)][:2]:
        item = {'id': _id(paper), 'url': paper.url, 'status': 'unavailable',
                'supplement': '未读取，是否存在未知'}
        result.append(item)
        try:
            identifier_query = ('DOI:' + item['id'][4:]) if item['id'].startswith('doi:') else ('EXT_ID:' + item['id'])
            search = _get(session, BASE + '/search', deadline, params={
                'query': identifier_query + ' AND SRC:MED',
                'format': 'json', 'resultType': 'core', 'pageSize': 2}).json()
            records = search['resultList']['result']
            records = [r for r in records if r.get('isOpenAccess') == 'Y' and (
                (item['id'].startswith('doi:') and str(r.get('doi', '')).casefold() == item['id'][4:]) or
                (not item['id'].startswith('doi:') and str(r.get('pmid', '')) == item['id']))]
            if len(records) != 1 or not re.fullmatch(r'PMC\d+', records[0].get('pmcid', '')):
                raise ValueError('OA PMCID unavailable')
            pmcid = records[0]['pmcid']
            xml = _get(session, BASE + '/' + quote(pmcid) + '/fullTextXML', deadline).content
            license_url, sections = _body(xml, pmcid)
            # Cap model input while recording exactly how much of each section was read.
            budget = 48000
            excerpt = []
            for section in sections:
                text = section['text'][:budget]
                if not text:
                    break
                excerpt.append({'heading': section['heading'], 'text': text,
                                'complete': len(text) == len(section['text'])})
                budget -= len(text)
            headings = ' '.join(s['heading'].lower() for s in excerpt)
            missing = [name for name, pattern in [('Methods', r'method|material'), ('Results', r'result'), ('Discussion', r'discussion')] if not re.search(pattern, headings)]
            item.update(pmcid=pmcid, license=license_url, read_sections=excerpt,
                        missing_or_unrecognized_sections=missing,
                        scope='仅提供的正文节选；图表、参考文献及补充材料未读取')
            prompt = ('只根据提供的正文节选输出 JSON: {"report": 字符串, "evidence": [{"quote": 原文逐字引用}]}。'
                      '区分事实、作者主张、推断与未知；不声称读过图表或补充材料。')
            output = _call(client, llm_params, prompt, excerpt, min(120, _remaining(deadline)))
            evidence = output.get('evidence')
            text = '\n'.join(s['text'] for s in excerpt)
            if not isinstance(output.get('report'), str) or not 1 <= len(output['report'].strip()) <= 12000 or not isinstance(evidence, list) or not evidence:
                raise ValueError('invalid reading report')
            if any(not isinstance(e, dict) or not isinstance(e.get('quote'), str) or not 8 <= len(e['quote']) <= 1000 or e['quote'] not in text for e in evidence):
                raise ValueError('untraceable evidence')
            item.update(status='success', report=output['report'], evidence=evidence)
        except Exception as error:
            item['reason'] = type(error).__name__
    return result


def _synthesize_week(papers, client, llm_params):
    """Generate up to three hypotheses from at least two completed public cards."""
    selected = [p for p in list(papers)[:20] if _eligible(p)]
    material = [{'id': _id(p), 'url': p.url, 'title': p.title, 'abstract': p.abstract} for p in selected]
    by_id = {p['id']: p for p in material}
    if len(by_id) < 2:
        return []
    prompt = ('只用所给摘要提出1至3个研究问题，输出JSON {"questions": [{"question":字符串，'
              '"existing_findings":字符串，"author_claims":字符串，"hypothesis":字符串，'
              '"conditions":字符串，"missing_evidence":字符串，'
              '"evidence": [{"id":文献id,"quote":摘要逐字原文}]}]}。'
              '每个问题需至少两篇不同文献的可核对引文；不能把不同资源中的相关性拼成机制，'
              '假设不是已验证结果；证据不足返回空数组。')
    try:
        output = _call(client, llm_params, prompt, material, 180)
        questions = output['questions']
        if not isinstance(questions, list) or len(questions) > 3:
            return []
        valid = []
        for item in questions:
            fields = ('question', 'existing_findings', 'author_claims', 'hypothesis', 'conditions', 'missing_evidence')
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in fields):
                continue
            evidence = item.get('evidence')
            if not isinstance(evidence, list) or not evidence:
                continue
            if any(not isinstance(e, dict) or str(e.get('id', '')) not in by_id or not isinstance(e.get('quote'), str) or len(e['quote']) < 8 or e['quote'] not in by_id[str(e['id'])]['abstract'] for e in evidence):
                continue
            if len({str(e['id']) for e in evidence}) < 2:
                continue
            item['evidence'] = [{'id': str(e['id']), 'quote': e['quote'], 'url': by_id[str(e['id'])]['url']} for e in evidence]
            valid.append(item)
        return valid
    except Exception:
        return []


def _bounded(function, seconds):
    # A slow upstream must never hold the no-mail job past its wall-clock budget.
    result = queue.Queue(maxsize=1)
    def run():
        try:
            result.put(function())
        except Exception:
            result.put([])
    threading.Thread(target=run, daemon=True).start()
    try:
        return result.get(timeout=seconds)
    except queue.Empty:
        return []


def extend(papers, client, llm_params, session=None):
    return _bounded(lambda: _extend(papers, client, llm_params, session), 120)


def synthesize_week(papers, client, llm_params):
    return _bounded(lambda: _synthesize_week(papers, client, llm_params), 180)
