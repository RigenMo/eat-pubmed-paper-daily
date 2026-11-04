import json

import pytest

from tests.canned_responses import make_sample_paper
from tests.test_executor import _delivery_executor


@pytest.fixture
def delivery(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config.recommendations.require_cards = False
    config.recommendations.max_papers = 5
    config.executor.debug = False
    config.executor.max_paper_num = 10
    papers = [make_sample_paper(title=f"Paper {i}", journal="Journal", journal_metrics={"jif": 12}) for i in range(5)]
    executor = _delivery_executor(config, papers)

    def reader(paper, *args):
        paper.abstract_zh = "Translated abstract"
        paper.reading_card = {"essence": ["Complete card"]}

    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_tldr", reader)
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_title_translation", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda *args, **kwargs: "HTML")
    return executor, papers, tmp_path / "outputs" / "delivery-receipt.json"


def test_preflight_and_failed_smtp_never_write_receipt(delivery, monkeypatch):
    executor, _, receipt = delivery
    assert executor.run(send_mail=False) == 5
    assert not receipt.exists()
    def fail(*args):
        raise RuntimeError("SMTP failed")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", fail)
    with pytest.raises(RuntimeError, match="SMTP failed"):
        executor.run()
    assert not receipt.exists()


@pytest.mark.parametrize("count,missing_abstract,passed", [(5, False, True), (3, False, False), (5, True, False)])
def test_successful_send_records_quality(delivery, monkeypatch, count, missing_abstract, passed):
    executor, papers, receipt_path = delivery
    executor.retrievers["stub"].retrieve_papers = lambda: papers[:count]
    if missing_abstract:
        def partial(paper, *args):
            paper.reading_card = {"essence": ["Card"]}
            paper.abstract_zh = None
        monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_tldr", partial)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: None)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    assert executor.run() == count
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["mail_sent"] is True
    assert receipt["run_id"] == "123" and receipt["run_attempt"] == "2"
    assert receipt["schema_version"] == 1 and len(receipt["date"]) == 10
    assert receipt["recorded_at_utc"].endswith("+00:00")
    assert receipt["paper_count"] == count and receipt["expected_cards"] == 5
    assert receipt["complete_cards"] == (0 if missing_abstract else count)
    assert receipt["quality_passed"] is passed
    assert len(receipt["papers"]) == count and receipt["papers"][0]["jif"] == 12


def test_receipt_write_failure_does_not_report_smtp_failure(delivery, monkeypatch):
    executor, _, _ = delivery
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.os.replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
    assert executor.run() == 5
