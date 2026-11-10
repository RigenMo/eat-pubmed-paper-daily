import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener

import hydra
from omegaconf import DictConfig

from daily_delivery_guard import SHANGHAI, SafeRedirect, api_pages, load_delivery_receipt
from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.card_store import load_ready, save_ready
from zotero_arxiv_daily.llm_client import FailoverClient
from zotero_arxiv_daily.reranker import get_reranker_cls
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.innovation import assess_innovation
from zotero_arxiv_daily.journal_metrics import enrich_journal_metrics
from zotero_arxiv_daily.paper_identity import filter_unseen, load_history, paper_keys, record_sent
from zotero_arxiv_daily.reading_preferences import effective_focus_mode, focus_evidence, order_candidates
from zotero_arxiv_daily.recommendations import verify_recommendations
from zotero_arxiv_daily.retriever.pubmed_retriever import PubMedRetriever, _publication_in_window
from zotero_arxiv_daily.utils import send_email

READ = re.compile(r'Reading candidate \d+; completed \d+/\d+: (https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?)')


def pmid(url):
    parts = urlsplit(url)
    match = re.fullmatch(r'/(?:pubmed/)?(\d+)/?', parts.path, flags=re.I) if (parts.hostname or '').casefold() in ('pubmed.ncbi.nlm.nih.gov', 'www.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov') else None
    if not match:
        raise ValueError('Invalid receipt PubMed URL')
    return match.group(1)


def today_history(repository, token, current_run, now):
    today = now.astimezone(SHANGHAI).date()
    start = datetime.combine(today - timedelta(days=1), datetime.min.time(), SHANGHAI).astimezone(timezone.utc)
    query = urlencode({'created': '>=' + start.strftime('%Y-%m-%dT%H:%M:%SZ')})
    base = f'https://api.github.com/repos/{repository}/actions'
    receipts, main_jobs, attempts = [], [], 0
    for workflow in ('main.yml', 'supplement.yml'):
        try:
            runs = list(api_pages(f'{base}/workflows/{workflow}/runs?{query}', token, 'workflow_runs'))
        except HTTPError as error:
            if workflow == 'supplement.yml' and error.code == 404:
                continue
            raise
        for run in runs:
            created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
            if workflow == 'supplement.yml' and created.astimezone(SHANGHAI).date() == today:
                attempts += 1
            run_id = str(run['id'])
            if run_id == str(current_run):
                continue
            receipt = load_delivery_receipt(repository, token, run_id)
            jobs = list(api_pages(f'{base}/runs/{run_id}/jobs?filter=all', token, 'jobs'))
            valid_receipt = receipt and receipt['date'] == today.isoformat()
            for job in jobs:
                for step in job.get('steps') or []:
                    delivery_step = ((workflow == 'main.yml' and job.get('name') == 'calculate-and-send' and step.get('name') == 'Run script') or (workflow == 'supplement.yml' and job.get('name') == 'supplement' and step.get('name') == 'Send missing cards'))
                    if not delivery_step or step.get('conclusion') not in ('success', 'failure', 'cancelled'):
                        continue
                    completed = datetime.fromisoformat(step['completed_at'].replace('Z', '+00:00'))
                    if completed.tzinfo is None:
                        raise ValueError('Missing completion timezone')
                    if completed.astimezone(SHANGHAI).date() != today or valid_receipt:
                        continue
                    log = read_job_log(repository, token, job['id'])
                    if workflow == 'supplement.yml':
                        safe = 'SUPPLEMENT_RETRY_SAFE true' in log and 'SUPPLEMENT_SMTP_STARTED' not in log
                        if not safe:
                            raise RuntimeError('Supplement send outcome unknown without receipt')
                    elif step['conclusion'] == 'success' or 'Sending email...' in log:
                        raise RuntimeError('Main send outcome unknown without receipt')
            if receipt and receipt['date'] == today.isoformat():
                receipts.append(receipt)
                if workflow == 'main.yml' and receipt['paper_count'] < 10:
                    main_jobs.extend(job['id'] for job in jobs if job['name'] == 'calculate-and-send')
    return receipts, main_jobs, max(1, attempts)


def read_job_log(repository, token, job_id):
    request = Request(f'https://api.github.com/repos/{repository}/actions/jobs/{job_id}/logs', headers={
        'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'})
    with build_opener(SafeRedirect()).open(request, timeout=30) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError('Job log too large')
    return data.decode('utf-8', errors='replace')


def read_candidates(repository, token, job_ids):
    found = []
    for job_id in job_ids:
        log = read_job_log(repository, token, job_id)
        found.extend(pmid(match) for match in READ.findall(log))
    return list(dict.fromkeys(found))


def write_summary(sent, cumulative, attempts):
    summary = {'complete': cumulative >= 10, 'sent': sent, 'cumulative': cumulative,
               'remaining': max(0, 10 - cumulative), 'attempt_count': attempts, 'retry_safe': True}
    output = Path('outputs/supplement-summary.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary), encoding='utf-8')
    print("SUPPLEMENT_RETRY_SAFE true", flush=True)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            stream.write(f"complete={str(summary['complete']).lower()}\ncumulative={cumulative}\nretry_safe=true\n")

def supplement(config, progress=None):
    if progress is None:
        progress = {}
    progress.update(retry_safe=False, cumulative=0, attempts=0)
    token, repository = os.environ['GITHUB_TOKEN'], os.environ['GITHUB_REPOSITORY']
    now = datetime.now(timezone.utc)
    deadline = time.monotonic() + max(0, float(getattr(config, "recommendations", {}).get("generation_budget_seconds", 1200)))
    receipts, job_ids, attempts = today_history(repository, token, os.environ['GITHUB_RUN_ID'], now)
    delivered = {}
    for receipt in receipts:
        for paper in receipt['papers']:
            identifier = pmid(paper['url'])
            if identifier in delivered:
                raise RuntimeError('Duplicate delivered paper in receipts')
            delivered[identifier] = paper
    if len(delivered) >= 10:
        write_summary(0, len(delivered), attempts)
        print(f'Supplement skipped: already delivered {len(delivered)}')
        return 0
    if attempts > 9:
        raise RuntimeError('Daily supplement attempt limit exceeded')
    if not receipts or not job_ids:
        raise RuntimeError('No confirmed main delivery available for supplementation')
    progress.update(retry_safe=True, cumulative=len(delivered), attempts=attempts)
    settings = getattr(config, "recommendations", {})
    mode = effective_focus_mode(settings)
    today = now.astimezone(SHANGHAI).date()
    history_path = Path(os.environ.get('SENT_PAPER_HISTORY_FILE', '.cache/sent-paper-history.json'))
    seen = set()
    if settings.get('exclude_sent', True):
        history = load_history(history_path, required=os.environ.get('GITHUB_ACTIONS', '').lower() == 'true')
        seen.update(history['entries'])
    for receipt in receipts:
        for paper in receipt['papers']:
            seen.update(paper_keys(paper))
    cached = []
    if settings.get('card_cache_enabled', False):
        try:
            lightweight = Executor.__new__(Executor)
            lightweight.config = config
            lightweight.include_path_patterns = normalize_path_patterns(config.zotero.include_path, 'include_path')
            lightweight.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, 'ignore_path')
            corpus = lightweight.filter_corpus(lightweight.fetch_zotero_corpus())
            if corpus:
                cached = filter_unseen(load_ready(config, corpus), seen)
        except Exception:
            cached = []
    selected = []
    if cached:
        # Receipts retain journal but not topic; retrieve delivered metadata to
        # preserve the same-day focus limit without constructing an Executor.
        try:
            retriever = PubMedRetriever(config)
            delivered_records = retriever._fetch_details(list(delivered))
            delivered_papers = [retriever.convert_to_paper(record) for record in delivered_records]
            by_id = {pmid(paper.url): paper for paper in delivered_papers if paper is not None}
            if any(identifier not in by_id for identifier in delivered):
                raise ValueError('Delivered topic metadata unavailable')
            focus_count = sum(bool(focus_evidence(by_id[identifier], mode)) for identifier in delivered)
            journal_counts = Counter(' '.join((paper['journal'] or '').casefold().split()) for paper in delivered.values())
            for paper in order_candidates(cached, mode):
                journal = ' '.join((paper.journal or '').casefold().split())
                is_focus = bool(focus_evidence(paper, mode))
                if journal_counts[journal] >= int(settings.get('max_per_journal', 3)) or (is_focus and focus_count >= 9):
                    continue
                selected.append(paper)
                journal_counts[journal] += 1
                focus_count += int(is_focus)
                if len(selected) >= 10 - len(delivered):
                    break
        except Exception:
            selected = []
    saved_selected = list(selected)
    try:
        if len(selected) < 10 - len(delivered):
            candidates = [item for item in read_candidates(repository, token, job_ids) if item not in delivered]
            if not candidates and not cached:
                raise RuntimeError('No unsent previously read candidates')
            executor = Executor(config)
            if isinstance(executor.openai_client, FailoverClient):
                executor.openai_client.deadline = deadline
            retriever = executor.retrievers['pubmed']
            records = retriever._fetch_details(list(delivered) + candidates)
            papers = [retriever.convert_to_paper(record) for record in records]
            papers = [paper for paper in papers if paper is not None]
            by_id = {pmid(paper.url): paper for paper in papers}
            if any(identifier not in by_id for identifier in delivered):
                raise RuntimeError('Cannot establish topic of all delivered papers')
            settings = config.recommendations
            mode = effective_focus_mode(settings)
            focus_count = sum(bool(focus_evidence(by_id[identifier], mode)) for identifier in delivered)
            journal_counts = Counter(' '.join((paper['journal'] or '').casefold().split()) for paper in delivered.values())
            today = now.astimezone(SHANGHAI).date()
            start = today - timedelta(days=int(settings.get('publication_days', 90)))
            eligible = [by_id[identifier] for identifier in candidates if identifier in by_id and
                        _publication_in_window(by_id[identifier].publication_date, start, today)]
            # Today's count is independent of the long-term ledger; only new candidates
            # are filtered, before journal lookups, verification and reading generation.
            history_path = Path(os.environ.get('SENT_PAPER_HISTORY_FILE', '.cache/sent-paper-history.json'))
            if settings.get('exclude_sent', True):
                history = load_history(history_path, required=os.environ.get('GITHUB_ACTIONS', '').lower() == 'true')
                seen = set(history['entries'])
            else:
                seen = set()
            for receipt in receipts:
                for paper in receipt['papers']:
                    seen.update(paper_keys(paper))
            eligible = filter_unseen(eligible, seen)
            if not eligible and not cached:
                raise RuntimeError('No unseen eligible candidates; no email sent')
            eligible = enrich_journal_metrics(eligible, config.journals)
            corpus = executor.filter_corpus(executor.fetch_zotero_corpus())
            if not corpus:
                raise RuntimeError('No eligible Zotero references')
            if executor.reranker is None:
                executor.reranker = get_reranker_cls(config.executor.reranker)(config)
            eligible = executor.reranker.rerank(eligible, corpus, keyword_query=None,
                                                favorite_journals=config.executor.get('favorite_journals', None))
            focus, library = [], []
            for paper in eligible:
                evidence = focus_evidence(paper, mode)
                if evidence:
                    paper.recommendation_matches = [evidence]
                    focus.append(paper)
                else:
                    library.append(paper)
            eligible = focus + verify_recommendations(library, executor.openai_client, config.llm,
                                                       batch_size=int(settings.get('batch_size', 5)))
            if settings.get('innovation_enabled', True):
                assess_innovation(eligible, executor.openai_client, config.llm)
            if cached:
                cached_keys = set().union(*(paper_keys(paper) for paper in cached))
                eligible = cached + [paper for paper in order_candidates(eligible, mode) if not paper_keys(paper) & cached_keys]
            else:
                eligible = order_candidates(eligible, mode)
            selected = []
            for paper in eligible:
                if len(selected) >= 10 - len(delivered):
                    break
                journal = ' '.join((paper.journal or '').casefold().split())
                is_focus = bool(focus_evidence(paper, mode))
                if journal_counts[journal] >= int(settings.get('max_per_journal', 3)) or (is_focus and focus_count >= 9):
                    continue
                if paper not in cached:
                    if time.monotonic() >= deadline:
                        continue
                    paper.generate_tldr(executor.openai_client, config.llm)
                    save_ready(paper, config, corpus)
                if not paper.reading_card or not paper.abstract_zh or paper.reader_status != 'success':
                    continue
                if paper not in cached and time.monotonic() < deadline:
                    paper.generate_title_translation(executor.openai_client, config.llm)
                selected.append(paper)
                journal_counts[journal] += 1
                focus_count += int(is_focus)
    except Exception as error:
        if not saved_selected:
            raise
        print(f"New-candidate processing failed ({type(error).__name__}); using saved cards", flush=True)
        selected = saved_selected
    if not selected:
        raise RuntimeError('No verified complete unsent paper; no email sent')
    count = len(selected)
    total = len(delivered) + count
    if datetime.now(timezone.utc).astimezone(SHANGHAI).date() != today:
        progress['retry_safe'] = False
        raise RuntimeError('Prepared supplement crossed the Shanghai day; no email sent')
    html = render_email(selected, summary=f'今日补充{count}篇，累计{total}/10。')
    progress['retry_safe'] = False
    print('SUPPLEMENT_SMTP_STARTED', flush=True)
    send_email(config, html, subject=f'Eat Paper Daily {today:%Y/%m/%d}（补充{count}篇，今日累计{total}/10）')
    if settings.get('exclude_sent', True):
        try:
            record_sent(history_path, selected, today)
        except Exception:
            print('SMTP accepted; local sent-paper history write failed. Reconcile from receipt.')
    receipt_written = False
    try:
        confirmed = datetime.now(timezone.utc)
        receipt = {'schema_version': 1, 'run_id': os.environ['GITHUB_RUN_ID'],
                   'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
                   'date': confirmed.astimezone(SHANGHAI).date().isoformat(),
                   'recorded_at_utc': confirmed.isoformat(), 'mail_sent': True,
                   'paper_count': count, 'complete_cards': count, 'expected_cards': count,
                   'quality_passed': True, 'day_expected_cards': 10, 'day_complete_cards': total,
                   'is_supplement': True,
                   'papers': [{'url': p.url, 'journal': p.journal, 'publication_date': p.publication_date,
                               'jif': (p.journal_metrics or {}).get('jif'),
                               'identity_keys': sorted(paper_keys(p))} for p in selected]}
        output = Path('outputs/delivery-receipt.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent, delete=False) as temp:
            json.dump(receipt, temp, ensure_ascii=False, indent=2)
            temp_path = temp.name
        os.replace(temp_path, output)
        receipt_written = True
        progress.update(retry_safe=True, cumulative=total)
    except Exception:
        print('SMTP accepted; receipt write failed. Do not retry without manual reconciliation.')
    if receipt_written:
        try:
            write_summary(count, total, attempts)
        except Exception:
            print('SMTP accepted; summary write failed. Do not retry without reconciliation.')
    print(f'Supplement sent: {count}; daily total: {total}/10')
    return count


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(config: DictConfig):
    progress = {}
    try:
        supplement(config, progress)
    except Exception:
        # Retry only if the receipt ledger is known and SMTP has not started.
        safe = progress.get('retry_safe', False)
        cumulative = progress.get('cumulative', 0)
        attempts = progress.get('attempts', 0)
        summary = {'complete': cumulative >= 10, 'sent': 0, 'cumulative': cumulative,
                   'remaining': max(0, 10 - cumulative), 'attempt_count': attempts,
                   'retry_safe': safe}
        try:
            output = Path('outputs/supplement-summary.json')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary), encoding='utf-8')
            if os.environ.get('GITHUB_OUTPUT'):
                with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
                    stream.write(f"complete={str(summary['complete']).lower()}\ncumulative={cumulative}\nretry_safe={str(safe).lower()}\n")
        except Exception:
            pass
        print(f'SUPPLEMENT_RETRY_SAFE {str(safe).lower()}', flush=True)
        raise

if __name__ == '__main__':
    main()




