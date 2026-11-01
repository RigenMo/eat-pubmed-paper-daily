"""Exercise real API generation without credentials or permission to send mail."""
from collections import Counter
from pathlib import Path
import json
import os

import hydra
from omegaconf import DictConfig
from openai import OpenAI

from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.journal_metrics import enrich_journal_metrics
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.retriever.pubmed_retriever import PubMedRetriever


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(config: DictConfig):
    mode = os.environ.get("PREFLIGHT_MODE", "automatic")
    output = Path("outputs/preflight") / mode
    output.mkdir(parents=True, exist_ok=True)

    if mode == "reader":
        # Two public papers from the user's failed mail: one invalid locator,
        # one overlength card. These are regression cases, not recommendations.
        retriever = PubMedRetriever(config)
        records = retriever._fetch_details(["42907583", "42816584"])
        papers = [retriever.convert_to_paper(record) for record in records]
        papers = [paper for paper in papers if paper is not None]
        if len(papers) != 2:
            raise RuntimeError("Reader regression needs both original public citations")
        papers = enrich_journal_metrics(papers, config.journals)
        client = OpenAI(
            api_key=config.llm.api.key, base_url=config.llm.api.base_url,
            timeout=float(config.llm.api.timeout),
        )
        for paper in papers:
            paper.generate_tldr(client, config.llm)
        html = render_email(
            papers, summary="真实故障论文的精读生成回归测试；此页不代表本次推荐，也没有发送邮件。"
        )
    else:
        executor = Executor(config)
        executor.run(send_mail=False)
        papers = getattr(executor, "last_papers", [])
        html = getattr(executor, "last_email_content", render_email([]))

    # Only generated reports are exported. Full Zotero records, source abstracts,
    # request prompts and API credentials are never written to an artifact/log.
    reports = []
    for paper in papers:
        reports.append({
            "title": paper.title, "url": paper.url, "journal": paper.journal,
            "first_author": paper.authors[0] if paper.authors else None,
            "publication_date": paper.publication_date,
            "publication_history": paper.publication_history,
            "journal_metrics": paper.journal_metrics,
            "abstract_zh": paper.abstract_zh, "reading_card": paper.reading_card,
            "reader_status": paper.reader_status,
            "card_characters": len(Paper._format_reading_card(paper.reading_card)) if paper.reading_card else None,
            "recommendation_matches": paper.recommendation_matches,
        })
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "mode": mode, "papers": len(papers),
        "complete_cards": sum(paper.reading_card is not None for paper in papers),
        "journals": dict(Counter(paper.journal for paper in papers)),
        "mail_sent": False,
    }
    print("PREFLIGHT_RESULT " + json.dumps(summary, ensure_ascii=False))
    if mode == "reader" and summary["complete_cards"] != 2:
        raise RuntimeError("The real reader regression did not produce both complete cards")


if __name__ == "__main__":
    main()
