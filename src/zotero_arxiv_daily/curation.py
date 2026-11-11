"""Build validated, unsent abstract reading cards without sending mail."""
import copy
import json
import os
import queue
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from omegaconf import open_dict

from .card_store import load_ready, save_ready
from .discovery import discover, sample_candidates
from .executor import Executor
from .journal_metrics import enrich_journal_metrics
from .paper_identity import filter_unseen, load_history, paper_keys
from .bibliographic_verification import verified_openalex, verify_openalex

TEXT_FIELDS = ('original_question', 'reframed_question', 'transfer_action', 'testable_question',
               'data_requirements', 'conditions', 'missing_evidence')
THOUGHT_FIELDS = ('scientific_contribution', 'thought_inspiration', 'resource_feasibility', 'key_quote', *TEXT_FIELDS)
LEVELS = {'high', 'medium', 'low', 'uncertain'}
RUBRIC = '''Evaluate each public abstract independently. Return JSON {"items":[...]} with the supplied id and:
scientific_contribution, thought_inspiration, resource_feasibility: each high, medium, low or uncertain.
Also supply original_question, reframed_question, key_quote, transfer_action, testable_question,
data_requirements, conditions, missing_evidence. Scientific contribution and thought inspiration are
independent of resource availability. Resource context is only that Gaofeng cohort, UK Biobank,
NHANES, mouse metagenomics and metabolomics may exist; variable availability, follow-up and pairing
are unverified. Do not assume access, results or suitability. key_quote must be a literal 8-200
character substring of this paper's abstract. Explain in Chinese, label original_question as 【摘要】
or 【作者主张】, reframed_question and transfer_action as 【推断】, testable question as 【假设】, and data, conditions and missing evidence as
【待核验】. Give specific actionable content, never claim priority or first discovery. Return all fields.'''


def validate_thought_card(card, abstract):
    if not isinstance(card, dict) or not isinstance(abstract, str):
        return False
    if any(card.get(field) not in LEVELS for field in THOUGHT_FIELDS[:3]):
        return False
    if any(card[field] not in {'high', 'medium'} for field in THOUGHT_FIELDS[:2]):
        return False
    quote = card.get('key_quote')
    if not isinstance(quote, str) or not 8 <= len(quote) <= 200 or quote not in abstract:
        return False
    for field in TEXT_FIELDS:
        value = card.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 1200:
            return False
        content = value.replace('【推断】', '').replace('【假设】', '').replace('【待核验】', '').replace('【摘要】', '').replace('【作者主张】', '').strip(' ：:。.')
        if content.casefold() in {'', 'unknown', 'uncertain', 'n/a', 'none', '未知', '不确定', '无', '待核验'}:
            return False
    labels = {'reframed_question': '【推断】', 'transfer_action': '【推断】', 'testable_question': '【假设】',
              'data_requirements': '【待核验】', 'conditions': '【待核验】', 'missing_evidence': '【待核验】'}
    return (any(label in card['original_question'] for label in ('【摘要】', '【作者主张】'))
            and all(label in card[field] for field, label in labels.items()))


def evaluate_batch(papers, client, model, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('curation deadline')
    material = [{'id': str(i), 'title': paper.title, 'abstract': paper.abstract} for i, paper in enumerate(papers)]
    response = client.chat.completions.create(model=model, timeout=min(180, remaining), max_tokens=6000,
        response_format={'type': 'json_object'}, messages=[{'role': 'system', 'content': RUBRIC},
        {'role': 'user', 'content': json.dumps(material, ensure_ascii=False)}])
    items = json.loads(response.choices[0].message.content)['items']
    if not isinstance(items, list):
        raise ValueError('invalid batch')
    by_id = {}
    duplicates = set()
    for item in items:
        if isinstance(item, dict):
            key = str(item.get('id'))
            if key in by_id:
                duplicates.add(key)
            by_id[key] = item
    approved = []
    for i, paper in enumerate(papers):
        item = by_id.get(str(i), {})
        if str(i) in duplicates or not validate_thought_card(item, paper.abstract):
            continue
        paper.thought_card = {field: item[field] for field in THOUGHT_FIELDS}
        # Only resource names are confirmed in this system. No field-level
        # access, follow-up, pairing or experimental conditions were supplied.
        paper.thought_card['resource_feasibility'] = 'uncertain'
        paper.innovation_assessment = {'type': 'uncertain', 'reason': '仅据摘要，未核实与既有研究的比较。',
                                       'evidence': '', 'scope': 'abstract'}
        paper._approval_basis = 'cognitive_transfer'
        approved.append(paper)
    return approved


def _bounded(function, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('curation deadline')
    result = queue.Queue(maxsize=1)
    def run():
        try:
            result.put((True, function()))
        except Exception as error:
            result.put((False, error))
    threading.Thread(target=run, daemon=True).start()
    try:
        ok, value = result.get(timeout=remaining)
    except queue.Empty:
        raise TimeoutError('curation deadline') from None
    if not ok:
        raise value
    return value


def _title(paper):
    return ' '.join((paper.title or '').casefold().split())


def curate(config, deadline=None, max_new=12):
    if not isinstance(max_new, int) or not 1 <= max_new <= 12:
        raise ValueError('max_new must be between 1 and 12')
    deadline = min(deadline, time.monotonic() + 2700) if deadline is not None else time.monotonic() + 2700
    counts = {'status': 'ok', 'stock_before': 0, 'stock_after': 0, 'discovered': 0, 'approved': 0, 'prepared': 0, 'stock_target': 20, 'target_met': False}
    try:
        executor = Executor(config)
        client = executor.openai_client
        if hasattr(client, 'deadline'):
            client.deadline = deadline
        all_corpus = _bounded(executor.fetch_zotero_corpus, deadline)
        corpus = executor.filter_corpus(all_corpus)
        if not corpus:
            raise ValueError('Zotero corpus unavailable')
        history = load_history(Path(os.environ.get('SENT_PAPER_HISTORY_FILE', '.cache/sent-paper-history.json')),
                               required=os.environ.get('GITHUB_ACTIONS', '').lower() == 'true')
        today = datetime.now(timezone(timedelta(hours=8))).date()
        seen = {key for key, day in history['entries'].items() if today - timedelta(days=180) <= datetime.fromisoformat(day).date() <= today}
        seen.update(key for paper in all_corpus for key in paper_keys(paper))
        seen.update(getattr(executor, 'zotero_read_keys', set()))
        read_titles = {_title(p) for p in all_corpus if _title(p)} | getattr(executor, 'zotero_read_titles', set())
        stock = [p for p in filter_unseen(load_ready(config, corpus), seen) if _title(p) not in read_titles]
        counts['stock_before'] = counts['stock_after'] = len(stock)
        if len(stock) >= 20:
            counts['target_met'] = True
            return counts
        journal_counts = Counter(' '.join((p.journal or '').casefold().split()) for p in stock)
        stock_keys = set().union(*(paper_keys(p) for p in stock)) if stock else set()
        stage = 'discovery'
        candidates, audit = _bounded(lambda: discover(per_channel=100), deadline)
        counts['discovered'] = len(candidates)
        counts['discovery'] = audit
        if not any(item.get('status') in ('ok', 'success', 'true_empty') for item in audit):
            raise RuntimeError('discovery unavailable')
        # Discovery promotes OpenAlex records only after independent PubMed verification.
        candidates = [p for p in filter_unseen(candidates, seen | stock_keys)
                      if p.source in ('pubmed', 'openalex') and _title(p) not in read_titles]
        counts['unseen_candidates'] = len(candidates)
        if not config.journals.get('enabled') or config.journals.get('mode') != 'easyscholar' or not config.journals.easyscholar.get('api_key'):
            raise ValueError('live journal metrics unavailable')
        stage = 'journal_metrics'
        candidates = _bounded(lambda: enrich_journal_metrics(copy.deepcopy(candidates), config.journals), deadline)
        counts['if_eligible'] = len(candidates)
        leads = [p for p in candidates if p.source == 'openalex']
        confirmed = _bounded(lambda: verify_openalex(leads, today, deadline=deadline), deadline) if leads else []
        counts['crossref_verified'] = len(confirmed)
        candidates = [p for p in candidates if p.source == 'pubmed'] + confirmed
        candidates.sort(key=lambda p: p.publication_date or '', reverse=True)
        candidates = sample_candidates(candidates, min(20, len(candidates))) if candidates else []
        llm = copy.deepcopy(config.llm)
        llm.generation_kwargs.max_tokens = min(int(llm.generation_kwargs.get('max_tokens', 8192)), 8192)
        with open_dict(llm.generation_kwargs):
            llm.generation_kwargs.timeout = 180
        consecutive_failures = 0
        batch_size = min(3, max_new)
        for offset in range(0, len(candidates), batch_size):
            if counts['prepared'] >= max_new or counts['stock_after'] >= 20 or time.monotonic() >= deadline:
                break
            batch = candidates[offset:offset + batch_size]
            try:
                stage = 'thought_review'
                approved = _bounded(lambda: evaluate_batch(batch, client, llm.generation_kwargs.model, deadline), deadline)
            except Exception as error:
                counts['status'] = 'stock_unavailable'
                counts['failure_stage'], counts['error_type'] = stage, type(error).__name__
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
                continue
            consecutive_failures = 0
            counts['approved'] += len(approved)
            for paper in approved:
                if counts['prepared'] >= max_new or counts['stock_after'] >= 20 or time.monotonic() >= deadline:
                    break
                journal = ' '.join((paper.journal or '').casefold().split())
                if not journal or journal_counts[journal] >= 3:
                    continue
                try:
                    stage = 'complete_reading_card'
                    _bounded(lambda: paper.generate_tldr(client, llm), deadline)
                    if save_ready(paper, config, corpus):
                        counts['prepared'] += 1
                        counts['stock_after'] += 1
                        journal_counts[journal] += 1
                except Exception:
                    continue
    except Exception as error:
        counts['status'] = 'stock_unavailable'
        counts['failure_stage'], counts['error_type'] = locals().get('stage', 'setup_or_library'), type(error).__name__
    if time.monotonic() >= deadline and counts['stock_after'] < 20:
        counts['status'] = 'stock_unavailable'
    counts['stock_target'] = 20
    counts['target_met'] = counts['stock_after'] >= 20
    if not counts['target_met'] and counts['status'] == 'ok':
        counts['status'] = 'stock_partial' if counts['stock_after'] else 'no_verified_stock'
    return counts
