import pytest
from tests.canned_responses import make_sample_paper
from tests.test_executor import _delivery_executor
from zotero_arxiv_daily.paper_identity import load_history


def test_smtp_success_records_and_next_run_filters(config, monkeypatch, tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'history.json'))
    config.recommendations.exclude_sent = True
    paper = make_sample_paper()
    monkeypatch.setattr('zotero_arxiv_daily.protocol.Paper.generate_tldr', lambda *args: None)
    monkeypatch.setattr('zotero_arxiv_daily.protocol.Paper.generate_title_translation', lambda *args: None)
    monkeypatch.setattr('zotero_arxiv_daily.executor.render_email', lambda *args, **kwargs: 'body')
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *args: None)
    assert _delivery_executor(config, [paper]).run() == 1
    assert load_history(tmp_path / 'history.json')['entries']
    assert _delivery_executor(config, [paper]).run() == 0


def test_smtp_failure_does_not_record(config, monkeypatch, tmp_path):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'history.json'
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(path))
    config.recommendations.exclude_sent = True
    monkeypatch.setattr('zotero_arxiv_daily.protocol.Paper.generate_tldr', lambda *args: None)
    monkeypatch.setattr('zotero_arxiv_daily.protocol.Paper.generate_title_translation', lambda *args: None)
    monkeypatch.setattr('zotero_arxiv_daily.executor.render_email', lambda *args, **kwargs: 'body')
    def fail(*args):
        raise RuntimeError('smtp failed')
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', fail)
    with pytest.raises(RuntimeError, match='smtp failed'):
        _delivery_executor(config, [make_sample_paper()]).run()
    assert not path.exists()


def test_cloud_missing_history_fails_before_retrieval_or_smtp(config, monkeypatch, tmp_path):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'missing.json'))
    config.recommendations.exclude_sent = True
    executor = _delivery_executor(config, [])
    executor.retrievers['stub'].retrieve_papers = lambda: pytest.fail('must not retrieve')
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *args: pytest.fail('must not send'))
    with pytest.raises(FileNotFoundError):
        executor.run()


def test_cloud_preflight_allows_missing_history(config, monkeypatch, tmp_path):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('SENT_PAPER_HISTORY_FILE', str(tmp_path / 'missing.json'))
    config.recommendations.exclude_sent = True
    assert _delivery_executor(config, []).run(send_mail=False) == 0
