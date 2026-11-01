"""Tests for zotero_arxiv_daily.construct_email: render_email, get_stars, get_block_html."""

from zotero_arxiv_daily.construct_email import render_email, get_stars, get_block_html, get_empty_html
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_render_email_falls_back_to_url_when_pdf_url_missing():
    """Papers without a resolvable PDF (e.g. PubMed entries lacking a DOI) must
    still link somewhere useful, not render a literal 'None' href."""
    paper = make_sample_paper(
        url="https://pubmed.ncbi.nlm.nih.gov/12345/", pdf_url=None, score=7.0, tldr="ok"
    )
    html = render_email([paper])
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/12345/"' in html
    assert 'href="None"' not in html


def test_render_email_shows_translated_title_when_present():
    paper = make_sample_paper(title_zh="示例论文标题", score=7.0, tldr="ok")
    html = render_email([paper])
    assert "示例论文标题" in html


def test_render_email_omits_translated_title_row_when_absent():
    paper = make_sample_paper(title_zh=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "示例论文标题" not in html


def test_render_email_shows_stars_for_high_score():
    paper = make_sample_paper(score=8.5, tldr="ok")
    html = render_email([paper])
    assert '<span class="full-star">' in html


def test_render_email_no_stars_for_low_score():
    paper = make_sample_paper(score=3.0, tldr="ok")
    html = render_email([paper])
    assert '<div class="star-wrapper">' not in html
    assert '<span class="full-star">' not in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "Summary", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html


def test_render_email_escapes_content_and_keeps_basic_markdown():
    paper = make_sample_paper(
        title="<script>alert('x')</script>",
        tldr="**重点** <b>不要执行</b>",
        score=7.0,
    )
    paper.abstract_zh = "## 摘要标题\n含有 <img src=x onerror=alert(1)>。"
    paper.reading_card = {
        "essence": ["**核心**", "第二行"],
        "design": {
            "objects_data": "对象与数据",
            "task_definitions": "任务与定义",
            "implementation": "实现与比较",
            "comparison_validation_scope": "验证与范围",
        },
        "evidence": [],
        "innovation": ["创新一", "创新二"],
        "criticism": ["批判一", "批判二", "批判三"],
        "self_test": "自测问题？",
    }

    html = render_email([paper])

    assert "<script>" not in html
    assert "&lt;script&gt;alert('x')&lt;/script&gt;" in html
    assert "&lt;b&gt;不要执行&lt;/b&gt;" not in html  # hidden together with the legacy TLDR
    assert "<strong>重点</strong>" not in html  # legacy TLDR is hidden by the structured report
    assert "<strong>核心</strong>" in html
    assert "<strong>摘要标题</strong>" in html
    assert "<br>" in html


def test_render_email_includes_chinese_abstract_and_reading_card_sections():
    paper = make_sample_paper(score=7.0, tldr="TLDR")
    paper.abstract_zh = "这是完整的中文摘要。"
    paper.material_scope = "纳入 2020—2025 年的人群研究。"
    paper.reading_card = {
        "essence": "研究要解决的问题",
        "design": "队列研究设计",
        "evidence": "主要结果与不确定性",
        "innovation": "方法学创新",
        "critique": "需要谨慎解释的局限",
        "selftest": "为什么该比较可以支持这个结论？",
    }

    html = render_email([paper])

    for text in (
        "一、原文摘要中文版",
        "这是完整的中文摘要。",
        "材料范围",
        "纳入 2020—2025 年的人群研究。",
        "二、精读卡",
        "1．本质",
        "2．设计",
        "3．关键证据",
        "4．创新",
        "5．批判",
        "自测问题",
    ):
        assert text in html


def test_render_email_shows_verified_journal_metric_badges_and_safe_source():
    paper = make_sample_paper(score=7.0, tldr="TLDR", journal="Nature Aging")
    paper.journal_metrics = {
        "jif": 25.0,
        "year": 2025,
        "source": "https://www.nature.com/nataging/journal-impact",
        "jcr_category": "Geriatrics & Gerontology",
        "jcr_quartile": "Q1",
        "jcr_year": 2025,
        "cas_category": {"big": "医学", "small": "老年医学"},
        "cas_quartile": {"big": "1区", "small": "1区"},
        "cas_year": 2025,
    }

    html = render_email([paper])

    assert "期刊指标" in html
    assert "JIF(影响因子)：25.0（2025）" in html
    assert "JCR" in html and "Geriatrics &amp; Gerontology" in html and "Q1" in html
    assert "中科院分区大类" in html and "中科院分区小类" in html
    assert 'href="https://www.nature.com/nataging/journal-impact"' in html
    assert "原文链接" in html
    assert ">PDF<" not in html


def test_render_email_marks_missing_metrics_unverified_without_inference():
    paper = make_sample_paper(score=7.0, tldr="TLDR", journal="Unknown Journal")
    paper.journal_metrics = {}

    html = render_email([paper])

    assert "JIF(影响因子)：未核实" in html
    assert "JCR分区：未核实" in html
    assert "中科院分区大类：未核实" in html
    assert "中科院分区小类：未核实" in html
    assert "Q1" not in html


def test_render_email_rejects_unsafe_article_and_metric_source_urls():
    paper = make_sample_paper(
        score=7.0,
        tldr="TLDR",
        url="javascript:alert(1)",
        pdf_url="javascript:alert(2)",
        journal="Nature Aging",
    )
    paper.journal_metrics = {"source": "javascript:alert(3)"}

    html = render_email([paper])

    assert "javascript:" not in html
    assert "原文链接（未核实）" in html


def test_render_email_renders_canonical_reading_card_in_order_without_legacy_tldr():
    paper = make_sample_paper(score=7.0, tldr="legacy TLDR must be hidden")
    paper.abstract_zh = "这是中文摘要。"
    paper.material_scope = "摘要与正文节选"
    paper.reading_card = {
        "material_scope": "摘要与正文节选",
        "essence": ["本质一", "本质二"],
        "design": {
            "objects_data": "对象与数据",
            "task_definitions": "任务与定义",
            "implementation": "实现与比较",
            "comparison_validation_scope": "验证与范围",
        },
        "evidence": [
            {
                "claim": "主张",
                "comparison": "比较",
                "result": "结果",
                "locator": "摘要第1句",
            }
        ],
        "innovation": ["创新一", "创新二"],
        "criticism": ["批判一", "批判二", "批判三"],
        "self_test": "自测问题？",
    }

    html = render_email([paper])

    headings = (
        "一、原文摘要中文版",
        "二、精读卡",
        "1．本质",
        "2．设计",
        "3．关键证据",
        "4．创新",
        "5．批判",
        "自测问题",
    )
    positions = [html.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "legacy TLDR must be hidden" not in html
    assert "{'objects_data'" not in html
    assert "['本质一', '本质二']" not in html
    assert "对象与数据：" in html
    assert "主张 → 比较 → 结果（定位：摘要第1句）" in html
    assert html.index("原文链接") < html.index("一、原文摘要中文版")


def test_render_email_generation_failure_keeps_original_under_nontranslated_heading():
    paper = make_sample_paper(
        score=7.0,
        tldr="生成失败：保留原始摘要，未完成中文翻译与精读卡。\nOriginal abstract",
    )
    paper.abstract = "Original abstract"
    paper.abstract_zh = None
    paper.reading_card = None

    html = render_email([paper])

    assert "生成失败" in html
    assert "摘要原文（未完成翻译）" in html
    assert "Original abstract" in html
    assert "原文摘要中文版" not in html


def test_render_email_always_shows_unverified_metric_badges_without_journal():
    paper = make_sample_paper(score=7.0, journal=None)
    html = render_email([paper])

    assert "JIF(影响因子)" in html
    assert "JCR分区" in html
    assert "中科院分区大类" in html
    assert "中科院分区小类" in html
    assert html.count("来源未核实") >= 4
    assert html.count("年份未核实") >= 4


def test_generated_report_is_rendered_once_with_program_material_scope():
    from tests.canned_responses import make_stub_openai_client

    paper = make_sample_paper(full_text=None)
    paper.generate_tldr(make_stub_openai_client(), {})

    html = render_email([paper])

    assert paper.reading_card is not None
    assert html.count(paper.abstract_zh) == 1
    assert html.count(paper.reading_card["self_test"]) == 1
    assert html.count("一、原文摘要中文版") == 1
    assert html.count("二、精读卡") == 1
    assert "仅据摘要（正文与补充材料未提供）" in html
    assert "<strong>TLDR" not in html
    assert "'objects_data'" not in html
    assert html.index("5．批判") < html.index(paper.reading_card["self_test"])


def test_easyscholar_values_show_all_cas_subjects_without_inventing_years():
    source = "https://www.easyscholar.cc/console/query"
    paper = make_sample_paper(journal="Nature Aging")
    paper.journal_metrics = {
        "jif": 25.0, "year": 2025, "jif_year": None, "source": source,
        "jcr_quartile": "Q1", "jcr_year": None, "jcr_source": source,
        "cas_quartile": {"big": "医学1区", "small": "细胞生物学1区/老年医学1区/神经科学1区。"},
        "cas_year": None, "cas_source": source,
        "metrics_provider": "EasyScholar", "queried_at": "2026-11-01",
    }

    html = render_email([paper])

    assert "25.0（年份未提供）" in html
    assert "学科未提供 / Q1（年份未提供）" in html
    assert "细胞生物学1区/老年医学1区/神经科学1区。" in html
    assert "医学1区（年份未提供）" in html
    assert "EasyScholar 查询日期：2026-11-01（UTC，非指标年份）" in html
    assert "25.0（2025）" not in html
