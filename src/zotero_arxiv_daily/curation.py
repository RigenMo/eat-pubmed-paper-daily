"""Build validated, unsent abstract reading cards without sending mail."""
import copy
import json
import hashlib
import os
import queue
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from omegaconf import open_dict

from .card_store import load_ready, save_ready, policy_fingerprint
from .discovery import discover, sample_candidates
from .executor import Executor
from .journal_metrics import enrich_journal_metrics
from .paper_identity import filter_unseen, load_history, paper_keys
from .bibliographic_verification import verified_openalex, verify_openalex
from .llm_client import BackgroundWorkPaused

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
【待核验】. These are internal screening notes, not extra email sections.
The audience works in epidemiology and bioinformatics. Ask one central question: what scientific
intuition drove this design, what specific knowledge or evidence gap would remain without it,
and how can that reasoning change an epidemiological OR bioinformatic research question?
Use original_question for the original uncertainty and design-grounded intuition, and
reframed_question for the concrete gap and how the study addresses it. Reconstruct reasoning
from the supplied evidence; do not invent the author's private motivation or claim historical
priority. A new measurement capability counts only when it opens a specific scientific question.
In transfer_action, map the original problem structure to a concrete epidemiological or
bioinformatic question: identify the corresponding entities, assumption or comparison being
changed, and what evidence would distinguish the proposed explanation from an alternative.
In testable_question state a falsifiable implication; in conditions and missing_evidence state
the bridge's assumptions and unresolved evidence. Missing from the abstract does not mean the
authors did not perform a test. Do not force both target disciplines or invent available fields.
An analogous sensor application, a disease/database/algorithm swap, shared terminology, or a
generic promise of better prediction is insufficient. High thought_inspiration requires a
specific, defensible transfer of scientific reasoning; high scientific_contribution requires a
specific knowledge/evidence increment supported by the abstract. If either chain cannot be
explained from available material, mark that dimension uncertain or low, rather than inventing
a bridge. Resource feasibility remains independent and must not lower thought value.
Return all fields.'''


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
    response = client.chat.completions.create(model=model, timeout=min(240, remaining), max_tokens=6000,
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
        # New cross-topic stock needs both a supported contribution and a strong
        # reasoning bridge. Cache validation remains compatible with saved work.
        if item['scientific_contribution'] != 'high' or item['thought_inspiration'] != 'high':
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


def curate(config, deadline=None, max_new=12, should_yield=None):
    if not isinstance(max_new, int) or not 1 <= max_new <= 12:
        raise ValueError('max_new must be between 1 and 12')
    deadline = min(deadline, time.monotonic() + 2700) if deadline is not None else time.monotonic() + 2700
    counts = {'status': 'ok', 'stock_before': 0, 'stock_after': 0, 'discovered': 0, 'approved': 0, 'prepared': 0, 'stock_target': 20, 'target_met': False, 'stock_count_verified': False, 'deferred': {}}
    def priority():
        if should_yield is not None and should_yield():
            raise BackgroundWorkPaused('Sender has priority')
    def defer(paper, minutes):
        until = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        counts['deferred'].update({key: until for key in paper_keys(paper)})
    try:
        priority()
        executor = Executor(config)
        client = executor.openai_client
        if hasattr(client, 'deadline'):
            client.deadline = deadline
        if hasattr(client, 'priority_check'):
            client.priority_check = should_yield
        all_corpus = _bounded(executor.fetch_zotero_corpus, deadline)
        corpus = executor.filter_corpus(all_corpus)
        if not corpus:
            raise ValueError('Zotero corpus unavailable')
        counts['policy'] = policy_fingerprint(config, corpus)
        deferred = set()
        try:
            control = json.loads(Path(os.environ.get('INVENTORY_CONTROL_FILE', '.cache/inventory-replenishment.json')).read_text(encoding='utf-8'))
            if control.get('policy') == counts['policy']:
                deferred = {key for key, stamp in control.get('deferred', {}).items()
                            if datetime.fromisoformat(stamp.replace('Z', '+00:00')) > datetime.now(timezone.utc)}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        history = load_history(Path(os.environ.get('SENT_PAPER_HISTORY_FILE', '.cache/sent-paper-history.json')),
                               required=os.environ.get('GITHUB_ACTIONS', '').lower() == 'true')
        counts['sent_ledger_digest'] = hashlib.sha256(json.dumps(history['entries'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        today = datetime.now(timezone(timedelta(hours=8))).date()
        seen = {key for key, day in history['entries'].items() if today - timedelta(days=180) <= datetime.fromisoformat(day).date() <= today}
        seen.update(key for paper in all_corpus for key in paper_keys(paper))
        seen.update(getattr(executor, 'zotero_read_keys', set()))
        read_titles = {_title(p) for p in all_corpus if _title(p)} | getattr(executor, 'zotero_read_titles', set())
        stock = [p for p in filter_unseen(load_ready(config, corpus), seen) if _title(p) not in read_titles]
        counts['stock_before'] = counts['stock_after'] = len(stock)
        counts['stock_count_verified'] = True
        if len(stock) >= 20:
            counts['target_met'] = True
            return counts
        journal_counts = Counter(' '.join((p.journal or '').casefold().split()) for p in stock)
        stock_keys = set().union(*(paper_keys(p) for p in stock)) if stock else set()
        stage = 'discovery'
        priority()
        candidates, audit = _bounded(lambda: discover(per_channel=100), deadline)
        counts['discovered'] = len(candidates)
        counts['discovery'] = audit
        if not any(item.get('status') in ('ok', 'success', 'true_empty') for item in audit):
            raise RuntimeError('discovery unavailable')
        # Discovery promotes OpenAlex records only after independent PubMed verification.
        candidates = [p for p in filter_unseen(candidates, seen | stock_keys | deferred)
                      if p.source in ('pubmed', 'openalex') and _title(p) not in read_titles]
        counts['unseen_candidates'] = len(candidates)
        if not config.journals.get('enabled') or config.journals.get('mode') != 'easyscholar' or not config.journals.easyscholar.get('api_key'):
            raise ValueError('live journal metrics unavailable')
        # Interleave the whole pool before the bounded journal lookups. Do not
        # spend a validation run's entire budget checking hundreds of journals.
        ordered, remaining = [], list(candidates)
        while remaining:
            group = sample_candidates(remaining, min(20, len(remaining)))
            if not group:
                break
            ordered.extend(group)
            chosen = {id(p) for p in group}
            remaining = [p for p in remaining if id(p) not in chosen]
        stage = 'journal_metrics'
        priority()
        candidate_target = min(32, max(4, max_new * 3))
        candidates = _bounded(lambda: enrich_journal_metrics(copy.deepcopy(ordered), config.journals, max_retained=candidate_target), deadline)
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
            priority()
            if counts['prepared'] >= max_new or counts['stock_after'] >= 20 or time.monotonic() >= deadline:
                break
            batch = candidates[offset:offset + batch_size]
            try:
                stage = 'thought_review'
                approved = _bounded(lambda: evaluate_batch(batch, client, llm.generation_kwargs.model, deadline), deadline)
            except BackgroundWorkPaused:
                raise
            except Exception as error:
                counts['status'] = 'stock_unavailable'
                counts['failure_stage'], counts['error_type'] = stage, type(error).__name__
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
                continue
            consecutive_failures = 0
            counts['approved'] += len(approved)
            approved_ids = {id(p) for p in approved}
            for paper in batch:
                if id(paper) not in approved_ids:
                    defer(paper, 120)
            for paper in approved:
                priority()
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
                    else:
                        defer(paper, 20)
                    priority()
                except BackgroundWorkPaused:
                    raise
                except Exception:
                    defer(paper, 20)
                    continue
    except BackgroundWorkPaused:
        counts['status'] = 'paused_for_sender'
        counts['pause_reason'] = getattr(should_yield, 'reason', 'sender_has_priority')
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
