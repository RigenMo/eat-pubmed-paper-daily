"""Tests for PubMedRetriever."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.retriever.pubmed_retriever import (
    PubMedRetriever,
    _parse_affiliation,
    _parse_pubmed_xml,
)
from tests.canned_responses import (
    SAMPLE_PUBMED_ESEARCH_RESPONSE,
    SAMPLE_PUBMED_MEDLINE_TEXT,
)


SAMPLE_PUBMED_XML = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>30000003</PMID>
      <Article>
        <ArticleTitle>A publication-date fixture</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2027</Year><Month>Jan</Month><Day>15</Day></PubDate>
          </JournalIssue>
          <Title>Journal Full Name</Title>
          <ISOAbbreviation>J Full Name</ISOAbbreviation>
        </Journal>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background section.</AbstractText>
          <AbstractText Label="RESULTS">Results section.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName><Initials>J</Initials>
            <AffiliationInfo><Affiliation>Department A, University A</Affiliation></AffiliationInfo>
          </Author>
          <Author><CollectiveName>Example Consortium</CollectiveName></Author>
        </AuthorList>
        <ELocationID EIdType="doi" ValidYN="Y">10.1234/example.1</ELocationID>
        <ArticleDate DateType="Electronic">
          <Year>2026</Year><Month>Oct</Month><Day>31</Day>
        </ArticleDate>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="received"><Year>2025</Year><Month>Nov</Month><Day>2</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="revised"><Year>2026</Year><Month>Jan</Month><Day>5</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="accepted"><Year>2026</Year><Month>Feb</Month><Day>8</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="epublish"><Year>2026</Year><Month>Oct</Month><Day>31</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="ppublish"><Year>2027</Year><Month>Jan</Month><Day>15</Day></PubMedPubDate>
        <PubMedPubDate PubStatus="entrez"><Year>2026</Year><Month>Nov</Month><Day>1</Day></PubMedPubDate>
      </History>
      <ArticleIdList>
        <ArticleId IdType="pubmed">30000003</ArticleId>
        <ArticleId IdType="doi">10.1234/example.1</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


SAMPLE_PUBMED_XML_WITH_PARTIAL_DATE = """\
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>30000004</PMID>
      <Article>
        <ArticleTitle>A seasonal publication-date fixture</ArticleTitle>
        <Journal>
          <JournalIssue><PubDate><Year>2024</Year><Season>Winter</Season></PubDate></JournalIssue>
          <Title>Seasonal Journal</Title>
          <ISOAbbreviation>Seasonal J</ISOAbbreviation>
        </Journal>
        <Abstract><AbstractText>Unstructured abstract.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData><History /></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _patch_ncbi(monkeypatch, esearch_result=None, medline_text=None):
    """Patch requests.get so esearch.fcgi/efetch.fcgi return canned data."""
    import requests

    esearch_result = esearch_result if esearch_result is not None else SAMPLE_PUBMED_ESEARCH_RESPONSE
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    medline_text = medline_text if medline_text is not None else SAMPLE_PUBMED_MEDLINE_TEXT.replace("TI  - ", f"DEP - {today}\nTI  - ")

    def _patched(url, params=None, timeout=None, **kwargs):
        resp = SimpleNamespace(raise_for_status=lambda: None)
        if "esearch.fcgi" in url:
            resp.json = lambda: esearch_result
        elif "efetch.fcgi" in url:
            resp.text = medline_text
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.pubmed_retriever.sleep", lambda _: None)


def test_pubmed_retrieve_end_to_end(config, monkeypatch):
    _patch_ncbi(monkeypatch)
    retriever = PubMedRetriever(config)
    retriever.set_pubmed_query('"dry eye"[tiab]')
    papers = retriever.retrieve_papers()

    assert len(papers) == 2
    titles = {p.title for p in papers}
    assert "A sample PubMed paper on AI diagnostics" in titles
    assert all(p.source == "pubmed" for p in papers)


def test_parse_pubmed_xml_preserves_citation_and_publication_history():
    records = _parse_pubmed_xml(SAMPLE_PUBMED_XML)

    assert len(records) == 1
    record = records[0]
    assert record["PMID"] == "30000003"
    assert record["TI"] == "A publication-date fixture"
    assert record["AB"] == ["BACKGROUND: Background section.", "RESULTS: Results section."]
    assert record["AU"] == ["Smith J", "Example Consortium"]
    assert record["AD"] == ["Department A, University A"]
    assert record["JT"] == "Journal Full Name"
    assert record["TA"] == "J Full Name"
    assert record["LID"] == "10.1234/example.1 [doi]"

    # ArticleDate electronic is the online-first publication date even though
    # the journal issue date is later; no date is padded to January 1.
    assert record["publication_date"] == "2026-10-31"
    assert [entry["event"] for entry in record["publication_history"]] == [
        "received",
        "revised",
        "accepted",
        "epublish",
        "publication",
        "entrez",
        "ppublish",
    ]
    assert record["publication_history"][0] == {
        "event": "received",
        "date": "2025-11-02",
        "source": "https://pubmed.ncbi.nlm.nih.gov/30000003/",
    }
    assert record["publication_history"][-1]["event"] == "ppublish"


def test_parse_pubmed_xml_preserves_partial_season_date_without_fabricating_day():
    record = _parse_pubmed_xml(SAMPLE_PUBMED_XML_WITH_PARTIAL_DATE)[0]

    assert record["publication_date"] == "2024 Winter"
    assert record["publication_history"] == [
        {
            "event": "publication",
            "date": "2024 Winter",
            "source": "https://pubmed.ncbi.nlm.nih.gov/30000004/",
        }
    ]


def test_pubmed_convert_to_paper_full_record(config):
    retriever = PubMedRetriever(config)
    raw = {
        "PMID": "30000001",
        "TI": "A sample PubMed paper",
        "AB": "Sample abstract text.",
        "AU": ["Smith J", "Doe A"],
        "AD": [
            "Department of Ophthalmology, Harvard Medical School, Boston, MA 02115, USA.",
            "Department of Ophthalmology, Harvard Medical School, Boston, MA 02115, USA.",
        ],
        "JT": "Nature Medicine",
        "LID": "10.1038/s41591-026-00001-1 [doi]",
    }
    paper = retriever.convert_to_paper(raw)

    assert paper.title == "A sample PubMed paper"
    assert paper.authors == ["Smith J", "Doe A"]
    assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/30000001/"
    assert paper.pdf_url == "https://doi.org/10.1038/s41591-026-00001-1"
    assert paper.journal == "Nature Medicine"
    # Duplicate AD lines must be deduplicated while preserving order
    assert paper.affiliations == ["Department of Ophthalmology, Harvard Medical School"]


def test_pubmed_convert_to_paper_skips_missing_title(config):
    retriever = PubMedRetriever(config)
    raw = {"PMID": "1", "AB": "abstract only, no title"}
    assert retriever.convert_to_paper(raw) is None


def test_pubmed_convert_to_paper_skips_missing_abstract(config):
    retriever = PubMedRetriever(config)
    raw = {"PMID": "1", "TI": "title only, no abstract"}
    assert retriever.convert_to_paper(raw) is None


def test_pubmed_convert_to_paper_no_affiliation(config):
    retriever = PubMedRetriever(config)
    raw = {"PMID": "2", "TI": "No affiliation paper", "AB": "abstract", "AU": "Lee K"}
    paper = retriever.convert_to_paper(raw)
    assert paper.affiliations is None
    assert paper.authors == ["Lee K"]


def test_pubmed_preserves_abstract_structure_and_attaches_audited_metrics(config):
    retriever = PubMedRetriever(config)
    raw = {
        "PMID": "3", "TI": "Structured abstract", "JT": "Nature Aging",
        "AB": ["BACKGROUND: Original background.", "RESULTS: 12 participants; 3.0 years."],
    }

    paper = retriever.convert_to_paper(raw)

    assert paper.abstract == "BACKGROUND: Original background.\n\nRESULTS: 12 participants; 3.0 years."
    assert paper.journal_metrics["jif"] == 25.0
    assert paper.journal_metrics["year"] == 2025
    assert paper.journal_metrics.get("jcr_quartile") is None
    assert paper.journal_metrics.get("cas_quartile") is None


def test_parse_affiliation_strips_postal_code_and_electronic_address():
    ad = "Department of Chemistry, MIT, Cambridge, MA 02139, USA. Electronic address: x@x.edu."
    assert _parse_affiliation(ad) == "Department of Chemistry, MIT"


def test_pubmed_fallback_escalates_when_strict_query_too_narrow(config, monkeypatch):
    """Strict query returns too few hits (< _MIN_RESULTS) -> escalate to broad query."""

    def _fake_fetch_pmids(self, query):
        if "strict" in query:
            return ["1"]  # below _MIN_RESULTS
        return [str(i) for i in range(20)]  # broad query: plenty of results

    monkeypatch.setattr(PubMedRetriever, "_fetch_pmids", _fake_fetch_pmids)
    monkeypatch.setattr(PubMedRetriever, "_fetch_details", lambda self, pmids: [{"pmids": pmids, "publication_date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}])

    retriever = PubMedRetriever(config)
    retriever.set_pubmed_query("strict query")
    retriever.set_pubmed_broad_query("broad query")
    result = retriever._retrieve_raw_papers()

    assert result[0]["pmids"] == [str(i) for i in range(20)]


def test_pubmed_no_query_mode5_uses_date_only_query(config, monkeypatch):
    """Zotero-only ranking still needs a recent PubMed candidate pool."""
    captured_queries = []

    monkeypatch.setattr(
        PubMedRetriever,
        "_fetch_pmids",
        lambda self, query: captured_queries.append(query) or ["1", "2"],
    )
    monkeypatch.setattr(
        PubMedRetriever,
        "_fetch_details",
        lambda self, pmids: [{"pmids": pmids, "publication_date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}],
    )

    retriever = PubMedRetriever(config)
    expected_query = retriever._date_filter()
    result = retriever._retrieve_raw_papers()

    assert captured_queries == [expected_query]
    assert result[0]["pmids"] == ["1", "2"]


def test_pubmed_date_filter_uses_utc_today_publication_upper_bound(config):
    retriever = PubMedRetriever(config)
    query = retriever._date_filter()
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")

    assert "[EDAT]" not in query
    assert "[PDAT]" in query
    assert "[EPDAT]" in query
    assert '"3000"' not in query
    assert f'"{today}"[PDAT]' in query


def test_automatic_zotero_publication_window_overrides_legacy_retrieval_days(config):
    config.search.mode = 5
    config.recommendations.enabled = True
    config.recommendations.publication_days = 30
    config.executor.retrieval_days = 1
    retriever = PubMedRetriever(config)
    assert retriever._publication_days == 30
    start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y/%m/%d")
    assert f'"{start}"[PDAT]' in retriever._date_filter()


def test_pubmed_dynamic_easyscholar_mode_disables_static_journal_gate(config):
    with open_dict(config):
        config.journals.enabled = True
        config.journals.mode = "easyscholar"
        config.journals.easyscholar.enabled = True
        config.journals.easyscholar.api_key = "test-key"

    retriever = PubMedRetriever(config)

    assert retriever._dynamic_jif is True
    assert retriever._journal_filter_enabled is False
    assert retriever.is_allowed_journal("Unlisted Journal") is True


def test_pubmed_theme_queries_are_balanced_and_deduplicated(config, monkeypatch):
    captured = []

    def _fake_fetch_pmids(self, query):
        captured.append((query, self._max_results))
        if "theme one" in query:
            return ["1", "2", "3"]
        return ["2", "4"]

    monkeypatch.setattr(PubMedRetriever, "_fetch_pmids", _fake_fetch_pmids)
    monkeypatch.setattr(
        PubMedRetriever,
        "_fetch_details",
        lambda self, pmids: [{"PMID": pmid, "publication_date": datetime.now(timezone.utc).strftime("%Y-%m-%d")} for pmid in pmids],
    )

    retriever = PubMedRetriever(config)
    retriever.set_max_results(4)
    retriever.set_pubmed_queries(["theme one", "theme two"])
    result = retriever._retrieve_raw_papers()

    assert [record["PMID"] for record in result] == ["1", "2", "4", "3"]
    assert [limit for _, limit in captured] == [2, 2]
    assert all("theme one" not in query or "[PDAT]" in query for query, _ in captured)


def test_pubmed_empty_configured_theme_queries_do_not_fall_back_to_date_only(config, monkeypatch):
    monkeypatch.setattr(
        PubMedRetriever,
        "_fetch_pmids",
        lambda self, query: pytest.fail("empty configured theme queries must not use date-only retrieval"),
    )

    retriever = PubMedRetriever(config)
    retriever.set_pubmed_queries([])

    assert retriever._retrieve_raw_papers() == []


def test_pubmed_no_query_keyword_mode_raises(config):
    """Keyword modes must not fall back to an unfiltered PubMed search."""
    from omegaconf import open_dict

    with open_dict(config):
        config.search.mode = 3

    retriever = PubMedRetriever(config)

    with pytest.raises(ValueError, match="PubMed query is required"):
        retriever._retrieve_raw_papers()


def test_pubmed_max_results_override(config):
    retriever = PubMedRetriever(config)
    assert retriever._max_results == config.source.pubmed.max_results
    retriever.set_max_results(10)
    assert retriever._max_results == 10


def test_pubmed_esearch_uses_valid_sort_value(config, monkeypatch):
    """'pub+date' is not a recognized NCBI sort schema (silently ignored by the
    API, falling back to relevance ranking). The correct value is 'pub_date'.
    """
    captured_params = {}

    def _fake_ncbi_get(url, params):
        captured_params.update(params)
        return SimpleNamespace(json=lambda: {"esearchresult": {"idlist": []}})

    import zotero_arxiv_daily.retriever.pubmed_retriever as pubmed_retriever

    monkeypatch.setattr(pubmed_retriever, "_ncbi_get", _fake_ncbi_get)
    monkeypatch.setattr(pubmed_retriever, "sleep", lambda _: None)

    retriever = PubMedRetriever(config)
    retriever._fetch_pmids("some query")

    assert captured_params["sort"] == "pub_date"


def test_pubmed_fetch_details_requests_xml_and_parses_dates(config, monkeypatch):
    captured_params = {}

    def _fake_ncbi_get(url, params):
        captured_params.update(params)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        year, month, day = today.split("-")
        xml = SAMPLE_PUBMED_XML.replace("<Year>2026</Year><Month>Oct</Month><Day>31</Day>", f"<Year>{year}</Year><Month>{month}</Month><Day>{day}</Day>")
        return SimpleNamespace(text=xml)

    import zotero_arxiv_daily.retriever.pubmed_retriever as pubmed_retriever

    monkeypatch.setattr(pubmed_retriever, "_ncbi_get", _fake_ncbi_get)
    monkeypatch.setattr(pubmed_retriever, "sleep", lambda _: None)

    records = PubMedRetriever(config)._fetch_details(["30000003"])

    assert captured_params["rettype"] == "abstract"
    assert captured_params["retmode"] == "xml"
    assert records[0]["publication_date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")


@pytest.mark.parametrize("response_text", ["<PubmedArticleSet />", "<PubmedArticleSet>"])
def test_pubmed_fetch_details_rejects_empty_or_invalid_xml(config, monkeypatch, response_text):
    import zotero_arxiv_daily.retriever.pubmed_retriever as pubmed_retriever

    monkeypatch.setattr(
        pubmed_retriever,
        "_ncbi_get",
        lambda url, params: SimpleNamespace(text=response_text),
    )
    monkeypatch.setattr(pubmed_retriever, "sleep", lambda _: None)

    with pytest.raises(ValueError):
        PubMedRetriever(config)._fetch_details(["30000003"])


def test_pubmed_esearch_appends_the_configured_journal_constraint(config, monkeypatch):
    """The allowlist is part of every ESearch term, including date-only terms."""
    captured_params = {}

    with open_dict(config):
        config.journals.enabled = True
        config.journals.mode = "whitelist"

    def _fake_ncbi_get(url, params):
        captured_params.update(params)
        return SimpleNamespace(json=lambda: {"esearchresult": {"idlist": []}})

    import zotero_arxiv_daily.retriever.pubmed_retriever as pubmed_retriever

    monkeypatch.setattr(pubmed_retriever, "_ncbi_get", _fake_ncbi_get)
    monkeypatch.setattr(pubmed_retriever, "sleep", lambda _: None)

    retriever = PubMedRetriever(config)
    retriever._fetch_pmids(retriever._date_filter())

    term = captured_params["term"]
    assert '"Nature Aging"[Journal]' in term
    assert '"Eur Heart J"[Journal]' in term
    assert term.count("[Journal]") == len(retriever._journal_query_names)
    assert captured_params["retmax"] == retriever._max_results


def test_pubmed_finalise_keeps_only_exact_allowlisted_journals(config, monkeypatch):
    """Unknown, missing, and unrelated journals are excluded after MEDLINE fetch."""
    with open_dict(config):
        config.journals.enabled = True
        config.journals.mode = "whitelist"

    records = [
        {"PMID": "1", "JT": "Nature aging"},
        {"PMID": "2", "JT": "Nat Aging"},
        {"PMID": "3", "JT": "Nature Medicine"},
        {"PMID": "4", "JT": "Unlisted Journal", "TA": "Eur Heart J"},
        {"PMID": "5", "JT": "Unknown Journal"},
        {"PMID": "6"},
    ]
    monkeypatch.setattr(PubMedRetriever, "_fetch_details", lambda self, pmids: records)

    kept = PubMedRetriever(config)._finalise([str(i) for i in range(1, 7)])

    assert [record["PMID"] for record in kept] == ["1", "2", "4"]


def test_pubmed_empty_enabled_shortlist_fails_closed(config, monkeypatch):
    """An enabled empty shortlist must not fall back to unrestricted ESearch."""
    with open_dict(config):
        config.journals.enabled = True
        config.journals.mode = "whitelist"
        config.journals.whitelist = []

    def _unexpected_ncbi_get(*args, **kwargs):
        pytest.fail("empty enabled shortlist must not issue an unrestricted query")

    import zotero_arxiv_daily.retriever.pubmed_retriever as pubmed_retriever

    monkeypatch.setattr(pubmed_retriever, "_ncbi_get", _unexpected_ncbi_get)
    assert PubMedRetriever(config)._fetch_pmids("some query") == []


def test_pubmed_debug_does_not_truncate_before_journal_filter_or_ranking(config, monkeypatch):
    """PubMed retrieval keeps the complete candidate batch in debug mode."""
    with open_dict(config):
        config.executor.debug = True

    pmids = [str(i) for i in range(10)]
    monkeypatch.setattr(PubMedRetriever, "_fetch_details", lambda self, received: [{"pmids": received}])

    result = PubMedRetriever(config)._finalise(pmids)

    assert result == [{"pmids": pmids}]


def test_publication_window_rejects_old_future_and_uncertain_dates():
    from datetime import date
    from zotero_arxiv_daily.retriever.pubmed_retriever import _publication_in_window

    start, today = date(2026, 10, 25), date(2026, 11, 2)
    assert _publication_in_window("2026-10-31", start, today)
    assert not _publication_in_window("2026-10-01", start, today)
    assert not _publication_in_window("2026-11-03", start, today)
    assert not _publication_in_window("2026-11", start, today)
    assert not _publication_in_window("2026", start, today)
    assert not _publication_in_window(None, start, today)
    assert _publication_in_window("2026-10", date(2026, 10, 1), today)


def test_direct_fetch_keeps_old_record_and_retrieval_filters_publication_dates(config, monkeypatch):
    from datetime import timedelta
    import zotero_arxiv_daily.retriever.pubmed_retriever as module

    today = datetime.now(timezone.utc).date()
    old = today - timedelta(days=int(config.executor.retrieval_days) + 2)
    future = today + timedelta(days=1)
    article = SAMPLE_PUBMED_XML.split("<PubmedArticle>", 1)[1].split("</PubmedArticle>", 1)[0]
    articles = []
    for pmid, date in [("1", old), ("2", future), ("3", today)]:
        replacement = f"<Year>{date.year}</Year><Month>{date.month}</Month><Day>{date.day}</Day>"
        body = article.replace("30000003", pmid).replace(
            "<Year>2026</Year><Month>Oct</Month><Day>31</Day>", replacement
        )
        articles.append("<PubmedArticle>" + body + "</PubmedArticle>")
    xml = "<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"
    monkeypatch.setattr(module, "_ncbi_get", lambda url, params: SimpleNamespace(text=xml))
    monkeypatch.setattr(module, "sleep", lambda _: None)
    monkeypatch.setattr(PubMedRetriever, "_fetch_pmids", lambda self, query: ["1", "2", "3"])
    retriever = PubMedRetriever(config)

    assert [r["PMID"] for r in retriever._fetch_details(["1", "2", "3"])] == ["1", "2", "3"]
    assert [r["PMID"] for r in retriever._retrieve_raw_papers()] == ["3"]
