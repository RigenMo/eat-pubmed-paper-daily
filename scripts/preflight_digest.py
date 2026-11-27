"""Exercise real API generation without credentials or permission to send mail."""
from collections import Counter
from pathlib import Path
import json
import os


if __name__ == "__main__" and os.environ.get("GITHUB_EVENT_NAME") == "push":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if "[easyscholar-fields-only]" in (event.get("head_commit") or {}).get("message", ""):
        import re
        import requests
        from urllib.parse import urlsplit
        key = os.environ.get("EASYSCHOLAR_API_KEY")
        if not key:
            print("JOURNAL_FIELDS_RESULT missing_existing_secret")
            raise SystemExit(1)
        endpoint = "https://www.easyscholar.cc/open/getPublicationRank"
        def structure(value, depth=0):
            if depth >= 4:
                return type(value).__name__
            if isinstance(value, list):
                return [structure(value[0], depth + 1)] if value else []
            if isinstance(value, dict):
                return {k: structure(v, depth + 1) for k, v in value.items()
                        if isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9_]{1,70}", k)
                        and key not in k}
            return type(value).__name__
        done = False
        for journal in ("Nature Metabolism", "Cell Metabolism"):
            try:
                resp = requests.get(endpoint, params={"secretKey": key, "publicationName": journal},
                                    allow_redirects=False, timeout=20)
                if resp.status_code != 200:
                    print("JOURNAL_FIELDS_RESULT", journal, "http_status", resp.status_code)
                    continue
                body = resp.json()
                if not isinstance(body, dict):
                    print("JOURNAL_FIELDS_RESULT", journal, "unexpected_json_type")
                    continue
                code = body.get("code")
                data = body.get("data")
                official = data.get("officialRank") if isinstance(data, dict) else None
                ranks = official.get("all") if isinstance(official, dict) else None
                payload = {"status": code if type(code) is int else None,
                           "root_structure": structure(body),
                           "rank_structure": structure(ranks),
                           "jif_type": type(ranks.get("sciif")).__name__ if isinstance(ranks, dict) else None,
                           "jif_is_number_only": (
                               bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", str(ranks.get("sciif")).strip()))
                               if isinstance(ranks, dict) else False)}
                print("JOURNAL_FIELDS_RESULT", journal, json.dumps(payload), flush=True)
                done = done or (code == 200 and isinstance(ranks, dict) and bool(ranks))
            except (ValueError, requests.RequestException):
                print("JOURNAL_FIELDS_RESULT", journal, "response_unavailable", flush=True)
        raise SystemExit(0 if done else 1)

import hydra
from omegaconf import DictConfig, OmegaConf
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
    failure = None

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
        try:
            executor.run(send_mail=False)
        except Exception as error:
            failure = error
        papers = getattr(executor, "last_papers", [])
        html = getattr(executor, "last_email_content", render_email(papers))
        audit = getattr(executor, "recommendation_audit", [])
        (output / "selection-audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
    reports = OmegaConf.to_container(OmegaConf.create(reports), resolve=True)
    (output / "reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "mode": mode, "papers": len(papers),
        "complete_cards": sum(paper.reading_card is not None for paper in papers),
        "journals": dict(Counter(paper.journal for paper in papers)),
        "mail_sent": False,
        "failure_type": type(failure).__name__ if failure else None,
    }
    print("PREFLIGHT_RESULT " + json.dumps(summary, ensure_ascii=False))
    if failure is not None:
        raise failure
    if mode == "automatic" and not summary["complete_cards"]:
        raise RuntimeError("No complete qualifying recommendation; positive end-to-end validation is incomplete")
    if mode == "reader" and summary["complete_cards"] != 2:
        raise RuntimeError("The real reader regression did not produce both complete cards")


if __name__ == "__main__":
    main()

