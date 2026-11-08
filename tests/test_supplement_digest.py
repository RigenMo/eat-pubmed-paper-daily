import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('supplement_digest', Path(__file__).resolve().parents[1] / 'scripts' / 'supplement_digest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def receipt(count):
    return {'papers': [{'url': f'https://pubmed.ncbi.nlm.nih.gov/{i}/', 'journal': 'J', 'publication_date': '2026-11-01', 'jif': 12} for i in range(1, count + 1)]}


def test_complete_day_skips_network_and_send(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(10)], [], 1))
    monkeypatch.setattr(module, 'send_email', lambda *args, **kwargs: pytest.fail('must not send'))
    assert module.supplement(SimpleNamespace()) == 0


def test_no_unsent_candidate_does_not_send(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(8)], [123], 1))
    monkeypatch.setattr(module, 'read_candidates', lambda *args: ['1', '2'])
    monkeypatch.setattr(module, 'send_email', lambda *args, **kwargs: pytest.fail('must not send'))
    with pytest.raises(RuntimeError, match='No unsent'):
        module.supplement(SimpleNamespace())


def test_duplicate_delivery_fails_closed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(1), receipt(1)], [], 1))
    with pytest.raises(RuntimeError, match='Duplicate'):
        module.supplement(SimpleNamespace())


@pytest.mark.parametrize('smtp_fails', [False, True])
def test_supplement_filters_prior_days_and_records_only_after_acceptance(config, monkeypatch, tmp_path, smtp_fails):
    import json
    from datetime import datetime, timedelta, timezone
    from tests.canned_responses import make_sample_paper
    from zotero_arxiv_daily.paper_identity import load_history, paper_keys, record_sent

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    for key, value in {'GITHUB_TOKEN': 'test', 'GITHUB_REPOSITORY': 'owner/repo', 'GITHUB_RUN_ID': '42', 'GITHUB_RUN_ATTEMPT': '1'}.items():
        monkeypatch.setenv(key, value)
    history_path = tmp_path / 'history.json'
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(history_path))
    today = datetime.now(timezone.utc).astimezone(module.SHANGHAI).date()
    old = make_sample_paper(source='pubmed', url='https://pubmed.ncbi.nlm.nih.gov/9/', pdf_url='https://doi.org/10.1000/old', publication_date=today.isoformat())
    alias = make_sample_paper(source='pubmed', url='https://pubmed.ncbi.nlm.nih.gov/99/', pdf_url='https://doi.org/10.1000/OLD', publication_date=today.isoformat())
    new = make_sample_paper(source='pubmed', url='https://pubmed.ncbi.nlm.nih.gov/10/', pdf_url='', journal='Other journal', publication_date=today.isoformat(), reading_card={'essence': ['Completed']}, abstract_zh='译文', reader_status='success', journal_metrics={'jif': 12})
    delivered = [make_sample_paper(source='pubmed', url=f'https://pubmed.ncbi.nlm.nih.gov/{i}/', pdf_url='', journal=f'J{i}', publication_date=today.isoformat()) for i in range(1, 9)]
    record_sent(history_path, [old], today - timedelta(days=1))
    prior_keys = set(load_history(history_path)['entries'])
    config.recommendations.exclude_sent = True
    config.recommendations.innovation_enabled = False
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(8)], [123], 1))
    monkeypatch.setattr(module, 'read_candidates', lambda *args: ['9', '99', '10'])
    retriever = SimpleNamespace(_fetch_details=lambda ids: delivered + [old, alias, new], convert_to_paper=lambda p: p)
    fake_executor = SimpleNamespace(retrievers={'pubmed': retriever}, openai_client=object(),
        fetch_zotero_corpus=lambda: [object()], filter_corpus=lambda c: c,
        reranker=SimpleNamespace(rerank=lambda papers, *args, **kwargs: papers))
    monkeypatch.setattr(module, 'Executor', lambda config: fake_executor)
    checked = []
    def enrich(papers, settings):
        checked.extend(papers)
        return papers
    monkeypatch.setattr(module, 'enrich_journal_metrics', enrich)
    monkeypatch.setattr(module, 'focus_evidence', lambda *args: None)
    monkeypatch.setattr(module, 'verify_recommendations', lambda papers, *args, **kwargs: papers)
    monkeypatch.setattr(module, 'order_candidates', lambda papers, mode: papers)
    monkeypatch.setattr(module, 'render_email', lambda *args, **kwargs: 'body')
    monkeypatch.setattr(type(new), 'generate_tldr', lambda *args: None)
    monkeypatch.setattr(type(new), 'generate_title_translation', lambda *args: None)
    sends = []
    def send(*args, **kwargs):
        sends.append(True)
        if smtp_fails:
            raise RuntimeError('SMTP failed')
    monkeypatch.setattr(module, 'send_email', send)
    if smtp_fails:
        with pytest.raises(RuntimeError, match='SMTP failed'):
            module.supplement(config)
        assert set(load_history(history_path)['entries']) == prior_keys
        assert not Path('outputs/delivery-receipt.json').exists()
    else:
        assert module.supplement(config) == 1
        result = json.loads(Path('outputs/delivery-receipt.json').read_text(encoding='utf-8'))
        assert result['day_complete_cards'] == 9
        assert result['papers'][0]['identity_keys'] == sorted(paper_keys(new))
        assert paper_keys(new) <= set(load_history(history_path)['entries'])
    assert checked == [new]
    assert len(sends) == 1


def test_receipt_pmid_ignores_url_variations():
    assert module.pmid('https://PUBMED.ncbi.nlm.nih.gov/123/?utm_source=email#top') == '123'
    assert module.pmid('https://www.ncbi.nlm.nih.gov/pubmed/123') == '123'




@pytest.mark.parametrize('log,blocked', [
    ('SUPPLEMENT_SMTP_STARTED\nSUPPLEMENT_RETRY_SAFE false', True),
    ('SUPPLEMENT_RETRY_SAFE true', False),
])
def test_previous_supplement_failure_requires_safe_marker(monkeypatch, log, blocked):
    from datetime import datetime, timezone
    now = datetime(2026, 11, 7, 4, tzinfo=timezone.utc)
    run = {'id': 11, 'created_at': '2026-11-07T03:00:00Z'}
    job = {'id': 22, 'name': 'supplement', 'steps': [
        {'name': 'Send missing cards', 'conclusion': 'failure', 'completed_at': '2026-11-07T03:30:00Z'}]}
    def pages(url, token, key):
        if '/workflows/main.yml/' in url:
            return []
        if '/workflows/supplement.yml/' in url:
            return [run]
        return [job]
    monkeypatch.setattr(module, 'api_pages', pages)
    monkeypatch.setattr(module, 'load_delivery_receipt', lambda *args: None)
    monkeypatch.setattr(module, 'read_job_log', lambda *args: log)
    if blocked:
        with pytest.raises(RuntimeError, match='outcome unknown'):
            module.today_history('owner/repo', 'test', '12', now)
    else:
        assert module.today_history('owner/repo', 'test', '12', now)[0] == []
