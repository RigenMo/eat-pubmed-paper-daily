"""No-mail public discovery experiment; optional bounded model evaluation."""
import argparse
import html
import json
import os
import sys
import time
import threading
import queue
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from zotero_arxiv_daily.discovery import discover, sample_candidates
from zotero_arxiv_daily.innovation import _SYSTEM as OLD_RUBRIC

NEW_RUBRIC = '''Additionally evaluate scientific_contribution, thought_inspiration, resource_feasibility independently as high, medium, low or uncertain. Resource context ONLY: Gaofeng cohort, UK Biobank, NHANES, mouse metagenomics and metabolomics exist; variables, follow-up and pairing are unknown. Do not reject contribution because resources do not fit. For each item add scientific_contribution, thought_inspiration, resource_feasibility, original_question, reframed_question, key_quote, transfer_action, testable_question, data_requirements, conditions, missing_evidence. Explicitly tag source facts, author claims, inference, and unverified assumptions in prose. Label transfer actions and testable questions as inference or hypothesis. key_quote must be a literal substring of the supplied abstract, 8-200 characters. Only abstract-level evidence; no priority claims. Return the old fields and these fields together for each id. No private source material is provided.'''
FIELDS = ('original_question', 'reframed_question', 'transfer_action', 'testable_question', 'data_requirements', 'conditions', 'missing_evidence')
LEVELS = {'high', 'medium', 'low', 'uncertain'}


def evaluate(papers, client, model, deadline):
    output = []
    for offset in range(0, len(papers), 5):
        batch = papers[offset:offset + 5]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        material = [{'id': str(i), 'title': p.title, 'abstract': p.abstract} for i, p in enumerate(batch)]
        try:
            response = client.chat.completions.create(model=model, timeout=min(60, remaining), response_format={'type': 'json_object'},
                messages=[{'role': 'system', 'content': OLD_RUBRIC + '\n' + NEW_RUBRIC}, {'role': 'user', 'content': json.dumps(material, ensure_ascii=False)}])
            items = json.loads(response.choices[0].message.content)['items']
            by_id = {str(item.get('id')): item for item in items if isinstance(item, dict)}
            if len(items) != len(batch) or len(by_id) != len(batch):
                raise ValueError('invalid batch')
            for i, paper in enumerate(batch):
                item = by_id.get(str(i), {})
                old_quote = item.get('evidence', '')
                kind = item.get('type')
                old_valid = kind in {'cognitive', 'capability', 'mixed', 'uncertain'} and isinstance(item.get('reason'), str) and bool(item['reason'].strip()) and len(item['reason']) <= 180 and isinstance(old_quote, str) and ((8 <= len(old_quote) <= 200 and (old_quote in paper.title or old_quote in paper.abstract)) or (not old_quote and kind == 'uncertain'))
                quote = item.get('key_quote')
                new_valid = all(item.get(field) in LEVELS for field in ('scientific_contribution', 'thought_inspiration', 'resource_feasibility')) and all(isinstance(item.get(field), str) and item[field].strip() for field in FIELDS) and isinstance(quote, str) and 8 <= len(quote) <= 200 and quote in paper.abstract
                eligible = new_valid and item['scientific_contribution'] in {'high', 'medium'} and item['thought_inspiration'] in {'high', 'medium'}
                new = {k: item.get(k) for k in ('scientific_contribution', 'thought_inspiration', 'resource_feasibility', 'key_quote', *FIELDS)} if new_valid else None
                paper.thought_card = new
                output.append({'pmid': getattr(paper, 'pmid', ''), 'old': {k: item.get(k) for k in ('type', 'reason', 'evidence')} if old_valid else None, 'new': new, 'new_eligible': bool(eligible), 'scope': 'abstract', 'old_valid': old_valid, 'new_valid': new_valid})
        except Exception as exc:
            output.extend({'pmid': getattr(p, 'pmid', ''), 'old': None, 'new': None, 'scope': 'abstract', 'error': type(exc).__name__} for p in batch)
    return output


def bounded(function, seconds):
    """Return unavailable on timeout without waiting for an external service."""
    results = queue.Queue(maxsize=1)
    def run():
        try:
            results.put(('ok', function()))
        except Exception as error:
            results.put((type(error).__name__, None))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        return results.get(timeout=max(0.01, seconds))
    except queue.Empty:
        return 'timeout', None


def choose_anchors(papers, corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    candidate_text = [(p.title or '') + ' ' + (p.abstract or '') for p in papers]
    corpus_text = [(c.title or '') + ' ' + (c.abstract or '') for c in corpus]
    if not corpus_text or not candidate_text:
        return
    matrix = TfidfVectorizer(stop_words='english', max_features=30000).fit_transform(corpus_text + candidate_text)
    similarity = matrix[len(corpus):] @ matrix[:len(corpus)].T
    for index, paper in enumerate(papers):
        row = similarity.getrow(index).toarray().ravel()
        top = sorted(range(len(corpus)), key=lambda i: (-row[i], i))[:3]
        paper.recommendation_matches = [{'anchor_title': corpus[i].title, 'anchor_abstract': corpus[i].abstract} for i in top]


def compare_old_gate(papers, executor, config, deadline):
    from zotero_arxiv_daily.recommendations import _batch_material, _request_batch, _parse_and_validate, _RecommendationValidationError
    status, corpus = bounded(lambda: executor.filter_corpus(executor.fetch_zotero_corpus()), min(45, deadline - time.monotonic()))
    if status != 'ok' or not corpus:
        return [{'status': 'unavailable', 'pass': None} for _ in papers]
    status, _ = bounded(lambda: choose_anchors(papers, corpus), min(30, deadline - time.monotonic()))
    if status != 'ok':
        return [{'status': 'unavailable', 'pass': None} for _ in papers]
    output = []
    for offset in range(0, len(papers), 5):
        batch = papers[offset:offset + 5]
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            output.extend({'status': 'unavailable', 'pass': None} for _ in batch)
            continue
        try:
            material, records = _batch_material(batch, offset)
            response = _request_batch(executor.openai_client, config.llm, material, timeout=min(60, remaining))
            validated = _parse_and_validate(response, records)
            output.extend({'status': 'valid', 'pass': validated[str(i)]['related']} for i in range(len(batch)))
        except _RecommendationValidationError:
            output.extend({'status': 'invalid', 'pass': None} for _ in batch)
        except Exception:
            output.extend({'status': 'unavailable', 'pass': None} for _ in batch)
    # Private anchors and verifier text are never copied into reports.
    for paper in papers:
        paper.recommendation_matches = None
    return output


def render_review(papers):
    sections = []
    labels = [('original_question', '原问题'), ('reframed_question', '重新提问'), ('key_quote', '摘要原文证据'), ('transfer_action', '迁移动作【推断】'), ('testable_question', '待检验问题【假设】'), ('data_requirements', '数据需求【待核验】'), ('conditions', '成立条件【待核验】'), ('missing_evidence', '缺失证据【待核验】')]
    for p in papers:
        if not p.thought_card:
            continue
        rows = ''.join('<dt>' + html.escape(label) + '</dt><dd>' + html.escape(str(p.thought_card.get(key, ''))) + '</dd>' for key, label in labels)
        sections.append('<article><h2>' + html.escape(p.title) + '</h2><a href="' + html.escape(p.url, quote=True) + '">PubMed</a><dl>' + rows + '</dl></article>')
    return '<!doctype html><meta charset="utf-8"><title>公开候选思想卡旁路</title><h1>公开候选思想卡旁路</h1><p>仅据摘要；迁移与假设待检验，资源字段、随访和配对情况未知。</p>' + ''.join(sections)


def main():
    started = time.monotonic()
    deadline = started + 360
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Directory for summary.json and local public-candidate detail')
    parser.add_argument('--limit', type=int, default=20, choices=range(1, 21))
    parser.add_argument('--model', help='Optional model; requires OPENAI_API_KEY unless --compare-zotero')
    parser.add_argument('--base-url')
    parser.add_argument('--compare-zotero', action='store_true', help='Use configured Zotero and LLM for the old gate on the same pool')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status, found = bounded(lambda: discover(per_channel=30), 85)
    papers, audit = found if status == 'ok' else ([], [{'status': status}])
    pool = sample_candidates(papers, args.limit)
    independent_metadata = 0
    if pool:
        from zotero_arxiv_daily.retriever.public_index import fetch_public_records
        status, records = bounded(lambda: fetch_public_records([getattr(p, 'pmid', '') for p in pool[:5]], max_requests=1), 25)
        independent_metadata = len(records) if status == 'ok' else 0
    config = None
    executor = None
    if args.compare_zotero or os.environ.get('EASYSCHOLAR_API_KEY'):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'config')):
            config = compose(config_name='default')
        # Match the deployed endpoint and credentials without relying on old
        # CUSTOM_CONFIG model/API defaults. No SMTP credentials are supplied.
        config.search.mode = 5
        config.zotero.user_id = os.environ.get('ZOTERO_ID', config.zotero.user_id)
        config.zotero.api_key = os.environ.get('ZOTERO_KEY', config.zotero.api_key)
        if os.environ.get('OPENAI_API_KEY'):
            config.llm.api.key = os.environ['OPENAI_API_KEY']
        if args.base_url:
            config.llm.api.base_url = args.base_url
        if args.model:
            config.llm.generation_kwargs.model = args.model
    if_status = 'unavailable'
    before_if = len(pool)
    if config is not None and os.environ.get('EASYSCHOLAR_API_KEY'):
        from zotero_arxiv_daily.journal_metrics import enrich_journal_metrics
        # A copy ensures a timed-out background lookup cannot mutate the reporting pool.
        import copy
        if_status, filtered = bounded(lambda: enrich_journal_metrics(copy.deepcopy(pool), config.journals), min(70, max(0.01, deadline-time.monotonic())))
        if if_status == 'ok':
            pool = filtered
    gates = [{'status': 'not_requested', 'pass': None} for _ in pool]
    if args.compare_zotero:
        from zotero_arxiv_daily.executor import Executor
        status, executor = bounded(lambda: Executor(config), min(20, max(0.01, deadline-time.monotonic())))
        if status == 'ok':
            gates = compare_old_gate(pool, executor, config, min(deadline - 100, time.monotonic() + 150))
        else:
            gates = [{'status': 'unavailable', 'pass': None} for _ in pool]
    evaluations = []
    if executor is not None:
        model = args.model or config.llm.generation_kwargs.model
        evaluations = evaluate(pool, executor.openai_client, model, deadline)
    elif args.model:
        if not os.environ.get('OPENAI_API_KEY'):
            parser.error('OPENAI_API_KEY is required for model evaluation')
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, max_retries=0) if args.base_url else OpenAI(max_retries=0)
        evaluations = evaluate(pool, client, args.model, deadline)
    for i, result in enumerate(evaluations):
        result['old_gate'] = gates[i]
    extra = sum(bool(result.get('new_eligible')) and gates[i]['pass'] is False for i, result in enumerate(evaluations))
    summary = {'experiment': 'public_shadow_same_pool', 'mail_sent': False, 'independent_metadata_complete': independent_metadata, 'retrieval': audit, 'pool': len(pool), 'discovery_channels': dict(Counter(c for p in pool for c in p.discovery_channels)), 'if_status': if_status, 'if_filtered': before_if - len(pool) if if_status == 'ok' else None, 'date_filtered': sum(x.get('date_filtered', 0) for x in audit), 'metadata_filtered': sum(x.get('metadata_filtered', 0) for x in audit), 'old_gate_pass': sum(x['pass'] is True for x in gates), 'old_gate_invalid': sum(x['status'] == 'invalid' for x in gates), 'old_gate_unavailable': sum(x['status'] in ('unavailable', 'not_requested') for x in gates), 'new_gate_eligible': sum(bool(x.get('new_eligible')) for x in evaluations), 'new_gate_invalid_or_unavailable': len(pool) - sum(bool(x.get('new_valid')) for x in evaluations), 'cross_topic_added': extra, 'elapsed_seconds': round(time.monotonic() - started, 2), 'daily_standard_met': False, 'interpretation': 'TF-IDF selects three local anchors for the existing verifier, not the full production reranker. Cross-topic added requires a valid old rejection and valid new eligibility. No daily delivery claim; IF, date, journal cap, history, cards and selection are not end-to-end verified.'}
    detail = {'scope': 'abstract', 'note': 'Public candidates and model text only; Zotero corpus, private quotes and prompts omitted.', 'pool': [{'pmid': getattr(p, 'pmid', ''), 'title': p.title, 'abstract': p.abstract, 'url': p.url, 'journal': p.journal, 'publication_date': p.publication_date, 'channels': p.discovery_channels, 'source_metadata': p.source_metadata, 'journal_metrics': p.journal_metrics} for p in pool], 'evaluations': evaluations}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'detail.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'review.html').write_text(render_review(pool), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
