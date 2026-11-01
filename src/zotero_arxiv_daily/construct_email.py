from collections.abc import Mapping
from html import escape
import math
import re
from typing import Any
from urllib.parse import urlparse

from .protocol import Paper


framework = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="margin:0; padding:16px; background:#ffffff; color:#333333; font-family:Arial, 'Microsoft YaHei', sans-serif;">
  <div style="width:100%; max-width:900px; margin:0 auto;">
    __CONTENT__
    <div style="padding:20px 4px 4px; color:#777777; font-size:12px; line-height:1.6;">
      To unsubscribe, remove your email in your Github Action setting.
    </div>
  </div>
</body>
</html>
"""


_SOURCE_BADGE_STYLES = {
    "pubmed": "background-color:#0066cc; color:#fff;",
    "arxiv": "background-color:#b31b1b; color:#fff;",
    "biorxiv": "background-color:#9b4dca; color:#fff;",
    "medrxiv": "background-color:#2a9d8f; color:#fff;",
}
_SOURCE_LABELS = {
    "pubmed": "PubMed",
    "arxiv": "arXiv",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
}


def _text_value(value: Any) -> str:
    """Return scalar text only; never expose Python container repr in email."""
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return ""
    return str(value)


def _render_rich_text(value: Any) -> str:
    """Escape text and render only the small Markdown subset used by the report."""
    text = _text_value(value)
    if not text:
        return ""

    rendered_lines = []
    for line in text.splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        line_text = heading.group(1) if heading else line
        rendered = escape(line_text, quote=False)
        rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
        if heading:
            rendered = f"<strong>{rendered}</strong>"
        rendered_lines.append(rendered)
    return "<br>".join(rendered_lines)


def _render_paragraphs(value: Any, paragraph_style: str = "") -> str:
    """Render scalar text as separate paragraphs, omitting empty lines."""
    text = _text_value(value)
    if not text:
        return ""
    style = (
        "margin:6px 0; font-size:14px; line-height:1.75; color:#333333;"
        + paragraph_style
    )
    paragraphs = []
    for line in text.splitlines():
        if line.strip():
            paragraphs.append(f'<p style="{style}">{_render_rich_text(line)}</p>')
    return "".join(paragraphs)


def _safe_url(value: Any) -> str:
    """Return an escaped HTTP(S) URL, or an empty string for unsafe input."""
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw or any(char in raw for char in "\r\n"):
        return ""
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return escape(raw, quote=True)


def _metric_value(metrics: Any, *keys: str) -> Any:
    if isinstance(metrics, Mapping):
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None
    if metrics is None:
        return None
    for key in keys:
        if hasattr(metrics, key):
            return getattr(metrics, key)
    return None


def _metric_text(value: Any, unknown: str = "未提供") -> str:
    text = _text_value(value).strip()
    return escape(text, quote=False) if text else unknown


def _metric_badge(
    label: str,
    value: str,
    source: Any = None,
) -> str:
    source_url = _safe_url(source)
    source_link = (
        f' <a href="{source_url}" style="margin-left:6px; color:#664d03; font-size:11px;">来源</a>'
        if source_url
        else '<span style="margin-left:6px; color:#8a6d3b; font-size:11px;">来源未核实</span>'
    )
    return (
        '<span style="display:inline-block; margin:2px 5px 2px 0; padding:5px 8px; '
        'border-radius:4px; background-color:#fff3cd; color:#664d03; font-size:12px; '
        'line-height:1.45; max-width:100%; box-sizing:border-box; white-space:normal; '
        'overflow-wrap:anywhere; word-break:break-word;">'
        f"<strong>{escape(label, quote=False)}：{value}</strong>{source_link}</span>"
    )


def _with_year(value: Any, year: Any) -> str:
    """Render a metric and append only a year that was actually supplied."""
    value_text = _metric_text(value, "未提供")
    year_text = _text_value(year).strip()
    if year_text:
        return f"{value_text}（{escape(year_text, quote=False)}）"
    return value_text


def _level_value(value: Any, level: str) -> Any:
    """Read a value from a canonical CAS ``{big, small}`` mapping."""
    if not isinstance(value, Mapping):
        return None
    keys = (level, "大类") if level == "big" else (level, "小类")
    for key in keys:
        if key in value:
            selected = value[key]
            if isinstance(selected, Mapping):
                return selected.get("value")
            return selected
    return None


def _build_journal_metrics_html(metrics: Any) -> str:
    """Render verified/unverified metric badges inside valid table cells."""
    cas_category = _metric_value(metrics, "cas_category")
    cas_quartile = _metric_value(metrics, "cas_quartile")
    cas_year = _metric_value(metrics, "cas_year")
    cas_source = _metric_value(metrics, "cas_source")
    cas_version = _text_value(_metric_value(metrics, "cas_version")).strip()

    jif = _metric_value(metrics, "jif")
    jif_year = _metric_value(metrics, "jif_year", "year")
    # ``source`` belongs to the overall impact-factor record. JCR and CAS use
    # their own explicit source fields and never inherit it.
    jif_source = _metric_value(metrics, "source")
    jcr_category = _metric_value(metrics, "jcr_category")
    jcr_quartile = _metric_value(metrics, "jcr_quartile")
    jcr_year = _metric_value(metrics, "jcr_year")
    jcr_source = _metric_value(metrics, "jcr_source")

    jcr_parts = []
    if _text_value(jcr_category).strip():
        jcr_parts.append(_metric_text(jcr_category, "未提供"))
    if _text_value(jcr_quartile).strip():
        jcr_parts.append(_metric_text(jcr_quartile, "未提供"))
    jcr_value = " / ".join(jcr_parts) or "未提供"
    if _text_value(jcr_year).strip():
        jcr_value += f"（{escape(_text_value(jcr_year).strip(), quote=False)}）"

    cas_badges = []
    has_cas_mapping = isinstance(cas_category, Mapping) or isinstance(cas_quartile, Mapping)
    has_cas_scalar = not has_cas_mapping and (
        cas_category is not None or cas_quartile is not None
    )
    if has_cas_scalar:
        scalar_parts = []
        if _text_value(cas_category).strip():
            scalar_parts.append(_metric_text(cas_category, "未提供"))
        if _text_value(cas_quartile).strip():
            scalar_parts.append(_metric_text(cas_quartile, "未提供"))
        scalar_value = " / ".join(scalar_parts) or "未提供"
        if _text_value(cas_year).strip():
            scalar_value += f"（{escape(_text_value(cas_year).strip(), quote=False)}）"
        cas_badges.append(_metric_badge("中科院分区", scalar_value, cas_source))
    for level, label in (
        ("big", "中科院分区大类"),
        ("small", "中科院分区小类"),
    ):
        if cas_version:
            label += f"（{cas_version}）"
        if has_cas_scalar:
            continue
        category = _level_value(cas_category, level)
        quartile = _level_value(cas_quartile, level)
        # EasyScholar's CAS labels already include the subject name and
        # partition. Preserve all small-subject labels without aggregation.
        cas_parts = []
        if _text_value(category).strip():
            cas_parts.append(_metric_text(category, "未提供"))
        if _text_value(quartile).strip():
            cas_parts.append(_metric_text(quartile, "未提供"))
        cas_value = " / ".join(cas_parts) or "未提供"
        if _text_value(cas_year).strip():
            cas_value += f"（{escape(_text_value(cas_year).strip(), quote=False)}）"
        cas_badges.append(
            _metric_badge(label, cas_value, cas_source)
        )

    jif_value = _with_year(jif, jif_year)
    badges = [
        _metric_badge("JIF(影响因子)", jif_value, jif_source),
        _metric_badge("JCR分区", jcr_value, jcr_source),
        *cas_badges,
    ]
    return (
        '<tr><td style="padding:5px 16px 8px; font-size:12px; line-height:1.5;">'
        '<div style="margin-bottom:3px; color:#555555;"><strong>期刊指标</strong></div>'
        + "".join(badges)
        + "</td></tr>"
    )


def get_empty_html() -> str:
    return (
        '<table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" '
        'style="font-family:Arial, \'Microsoft YaHei\', sans-serif; border:1px solid #ddd; '
        'border-radius:8px; background-color:#f9f9f9;">'
        '<tr><td style="padding:16px; font-size:20px; font-weight:bold; color:#333333;">'
        "本次没有符合筛选条件的文献"
        "<br><span style=\"font-size:14px; font-weight:normal; color:#666666;\">"
        "未放宽主题相关性或期刊条件来补足数量</span>"
        "</td></tr></table>"
    )


def _build_source_badge(source: str | None, journal: str | None) -> str:
    """Return an escaped source badge and optional journal name."""
    source_text = _text_value(source).strip()
    if not source_text:
        return ""
    source_key = source_text.lower()
    style = _SOURCE_BADGE_STYLES.get(source_key, "background-color:#555; color:#fff;")
    label = _SOURCE_LABELS.get(source_key, source_text.upper())
    badge = (
        f'<span style="{style} border-radius:3px; padding:2px 7px; '
        f'font-size:12px; font-weight:bold;">{escape(label, quote=False)}</span>'
    )
    journal_text = _text_value(journal).strip()
    if journal_text:
        badge += (
            f'&nbsp;<span style="color:#555; font-style:italic; font-size:12px;">'
            f"{escape(journal_text, quote=False)}</span>"
        )
    return badge


def _section_row(heading: str, content: str) -> str:
    return (
        '<tr><td style="padding:7px 16px 2px; font-size:14px; line-height:1.7; overflow-wrap:anywhere;">'
        '<div style="margin:0 0 3px; color:#2c5f8a; font-size:15px;">'
        f"<strong>{escape(heading, quote=False)}</strong></div>"
        f"{content}"
        "</td></tr>"
    )


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if _text_value(item).strip()]


def _build_design_html(design: Any) -> str:
    if isinstance(design, Mapping):
        fields = (
            ("对象与数据", "objects_data"),
            ("任务与定义", "task_definitions"),
            ("实现与比较", "implementation"),
            ("验证与范围", "comparison_validation_scope"),
        )
        paragraphs = []
        for label, key in fields:
            value = design.get(key)
            if value is None:
                continue
            rendered = _render_rich_text(value)
            if rendered:
                paragraphs.append(
                    '<p style="margin:6px 0; font-size:14px; line-height:1.75; color:#333333;">'
                    f"<strong>{label}：</strong> {rendered}</p>"
                )
        return "".join(paragraphs)
    return ""


def _build_evidence_html(evidence: Any) -> str:
    evidence = evidence if isinstance(evidence, list) else []
    paragraphs = []
    for item in evidence:
        if isinstance(item, Mapping):
            claim = _render_rich_text(item.get("claim"))
            comparison = _render_rich_text(item.get("comparison"))
            result = _render_rich_text(item.get("result"))
            locator = _render_rich_text(item.get("locator"))
            if not any((claim, comparison, result, locator)):
                continue
            rendered = f"{claim} → {comparison} → {result}（定位：{locator}）"
        else:
            continue
        if rendered:
            paragraphs.append(
                '<p style="margin:6px 0; font-size:14px; line-height:1.75; color:#333333;">'
                f"{rendered}</p>"
            )
    if not paragraphs:
        paragraphs.append(
            '<p style="margin:6px 0; font-size:14px; line-height:1.75; color:#666666;">'
            "【未报告/材料未提供】</p>"
        )
    return "".join(paragraphs)


def _build_reading_report_html(
    abstract_zh: Any,
    reading_card: Any,
    material_scope: Any,
) -> str:
    """Render the canonical report without falling back to container reprs."""
    if not isinstance(reading_card, Mapping) or not reading_card or not _text_value(abstract_zh).strip():
        return ""

    scope = material_scope
    if not _text_value(scope).strip():
        scope = reading_card.get("material_scope")
    scope_html = ""
    if _text_value(scope).strip():
        scope_html = _render_paragraphs(f"材料范围：{_text_value(scope).strip()}")

    essence = _string_items(reading_card.get("essence"))
    design = reading_card.get("design")
    evidence = reading_card.get("evidence")
    innovation = _string_items(reading_card.get("innovation"))
    criticism = _string_items(reading_card.get("criticism"))
    self_test = reading_card.get("self_test")

    rows = [
        _section_row("一、原文摘要中文版", _render_paragraphs(abstract_zh)),
        _section_row("二、精读卡", scope_html),
        _section_row(
            "1．本质",
            "".join(_render_paragraphs(item) for item in essence)
            or _render_paragraphs("【未报告/材料未提供】"),
        ),
        _section_row(
            "2．设计",
            _build_design_html(design) or _render_paragraphs("【未报告/材料未提供】"),
        ),
        _section_row("3．关键证据", _build_evidence_html(evidence)),
        _section_row(
            "4．创新",
            "".join(_render_paragraphs(item) for item in innovation)
            or _render_paragraphs("【未报告/材料未提供】"),
        ),
        _section_row(
            "5．批判",
            "".join(_render_paragraphs(item) for item in criticism)
            or _render_paragraphs("【未报告/材料未提供】"),
        ),
        _section_row(
            "自测问题",
            _render_paragraphs(self_test) or _render_paragraphs("【未报告/材料未提供】"),
        ),
    ]
    return "".join(rows)


def _is_generation_failure(tldr: Any) -> bool:
    return _text_value(tldr).lstrip().startswith("生成失败")


def _build_fallback_rows(
    tldr: Any, original_abstract: Any, abstract_zh: Any, material_scope: Any = None
) -> str:
    """Render legacy TLDR or an explicit non-translated fallback."""
    original = _text_value(original_abstract).strip()
    if _text_value(abstract_zh).strip():
        return (
            _section_row("一、原文摘要中文版", _render_paragraphs(abstract_zh))
            + _section_row("二、精读卡", _render_paragraphs("精读卡未生成；摘要译文已保留。"))
        )

    if _is_generation_failure(tldr):
        failure = _render_paragraphs(_text_value(tldr).splitlines()[0])
        if material_scope:
            failure += _render_paragraphs(f"材料范围：{_text_value(material_scope)}")
        original_html = _render_paragraphs(original or "【未报告/材料未提供】摘要原文未提供。")
        return (
            _section_row("生成失败", failure)
            + _section_row("摘要原文（未完成翻译）", original_html)
        )

    legacy = _render_paragraphs(tldr)
    if legacy:
        return _section_row("TLDR", legacy)
    if original:
        return _section_row("摘要原文（未完成翻译）", _render_paragraphs(original))
    return ""


_PUBLICATION_EVENT_LABELS = {
    "received": "投稿",
    "submitted": "投稿",
    "submission": "投稿",
    "投稿": "投稿",
    "revised": "修回",
    "revision": "修回",
    "修回": "修回",
    "accepted": "接收",
    "accept": "接收",
    "接收": "接收",
    "aheadofprint": "在线发表",
    "epublish": "在线发表",
    "electronic": "在线发表",
    "online": "在线发表",
    "onlinefirst": "在线发表",
    "onlinepublished": "在线发表",
    "在线发表": "在线发表",
    "ppublish": "卷期发表",
    "print": "卷期发表",
    "printpublished": "卷期发表",
    "卷期发表": "卷期发表",
    "publication": "发表",
    "发表": "发表",
}
_PUBLICATION_INDEX_EVENTS = {
    "entrez",
    "medline",
    "medlineindex",
    "medlineindexed",
    "pubmed",
    "pubmednotmedline",
    "pmc",
    "pmcrelease",
    "pmcreleased",
}


def _publication_event_label(event: Any) -> str:
    raw = _text_value(event).strip()
    if not raw:
        return ""
    normalized = raw.casefold().replace(" ", "").replace("_", "").replace("-", "")
    if normalized in _PUBLICATION_INDEX_EVENTS or normalized.startswith(
        ("entrez", "medline", "pubmed", "pmc")
    ):
        return ""
    return _PUBLICATION_EVENT_LABELS.get(raw, _PUBLICATION_EVENT_LABELS.get(normalized, ""))


def _build_publication_html(
    publication_date: Any,
    publication_history: Any,
    *,
    show_missing: bool = False,
) -> str:
    """Render actual dates only; preserve the chronology supplied upstream."""
    publication_text = _text_value(publication_date).strip()
    publication_source_url = ""
    history_items = []
    if isinstance(publication_history, list):
        for item in publication_history:
            if not isinstance(item, Mapping):
                continue
            date_text = _text_value(item.get("date")).strip()
            label = _publication_event_label(item.get("event"))
            if not date_text or not label:
                continue
            source_url = _safe_url(item.get("source"))
            # Keep the publication date in the chronological history too.
            if label == "发表" and publication_text and date_text == publication_text:
                publication_source_url = publication_source_url or source_url
            date_html = escape(date_text, quote=False)
            if source_url:
                date_html += (
                    f' <a href="{source_url}" style="color:#2c5f8a; font-size:11px;">来源</a>'
                )
            history_items.append(
                f'<span style="display:inline-block; margin:0 10px 3px 0;">'
                f"<strong>{escape(label, quote=False)}</strong>：{date_html}</span>"
            )

    rows = []
    if publication_text:
        publication_date_html = escape(publication_text, quote=False)
        if publication_source_url:
            publication_date_html += (
                f' <a href="{publication_source_url}" style="color:#2c5f8a; font-size:11px;">来源</a>'
            )
        rows.append(
            '<div style="margin:2px 0; color:#555555;">'
            f"<strong>首次发表/发表日期：</strong>{publication_date_html}</div>"
        )
    if history_items:
        rows.append(
            '<div style="margin:2px 0; color:#555555;">'
            "<strong>出版历程：</strong>"
            + ' <span aria-hidden="true">→</span> '.join(history_items)
            + "</div>"
        )
    if not rows and show_missing:
        rows.append(
            '<div style="margin:2px 0; color:#777777;">发表日期未提供</div>'
        )
    if not rows:
        return ""
    return (
        '<tr><td style="padding:2px 16px 4px; font-size:12px; line-height:1.5; '
        'overflow-wrap:anywhere; white-space:normal;">'
        + "".join(rows)
        + "</td></tr>"
    )


def _build_recommendation_html(matches: Any) -> str:
    """Render verified Zotero relationship explanations without raw scores."""
    if not isinstance(matches, list):
        return ""

    entries = []
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        reason = _text_value(match.get("reason")).strip()
        title = _text_value(match.get("anchor_title")).strip()
        anchor_url = _safe_url(match.get("anchor_url"))
        if not reason and not title:
            continue

        source_title = _render_rich_text(title) if title else "Zotero 文献"
        if anchor_url:
            source_title = f'<a href="{anchor_url}">{source_title}</a>'
        reason_html = _render_rich_text(reason)
        if reason_html and title:
            reason_html += f"（来源：{source_title}）"
        elif reason_html:
            reason_html += f"（来源：{source_title}）" if anchor_url else ""
        else:
            reason_html = source_title
        entries.append(
            '<p style="margin:4px 0; font-size:13px; line-height:1.6; '
            'overflow-wrap:anywhere;">'
            + reason_html
            + "</p>"
        )

    if not entries:
        return ""
    return _section_row("与 Zotero 的关系", "".join(entries))


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str = None,
    source: str = None,
    journal: str = None,
    stars: str = "",
    title_zh: str = None,
    abstract_zh: str = None,
    reading_card: Any = None,
    material_scope: Any = None,
    journal_metrics: Any = None,
    original_abstract: str = None,
    publication_date: Any = None,
    publication_history: Any = None,
    recommendation_matches: Any = None,
    show_missing_date: bool = False,
):
    """Build one valid table card; the positional legacy signature is kept."""
    source_badge = _build_source_badge(source, journal)
    source_row = (
        '<tr><td style="padding:4px 16px 6px;">' + source_badge + "</td></tr>"
        if source_badge
        else ""
    )
    title_zh_text = _text_value(title_zh).strip()
    title_zh_row = (
        '<tr><td style="font-size:15px; color:#555555; padding:2px 16px; line-height:1.6;">'
        + _render_rich_text(title_zh_text)
        + "</td></tr>"
        if title_zh_text
        else ""
    )

    # If the primary URL is unsafe, the caller may still have supplied a safe
    # fallback in pdf_url; direct callers get the explicit unverified label.
    article_url = _safe_url(pdf_url)
    article_link = (
        f'<a href="{article_url}" style="display:inline-block; text-decoration:none; '
        'font-size:14px; font-weight:bold; color:#ffffff; background-color:#d9534f; '
        'padding:8px 16px; border-radius:4px;">原文链接</a>'
        if article_url
        else '<span style="font-size:14px; color:#777777;">原文链接（未核实）</span>'
    )

    structured_report = _build_reading_report_html(abstract_zh, reading_card, material_scope)
    report_rows = structured_report or _build_fallback_rows(tldr, original_abstract, abstract_zh, material_scope)
    # Keep the metric badges visible even when values are absent; ``未提供``
    # makes missing metadata explicit without inferring a year or category.
    metrics_row = _build_journal_metrics_html(journal_metrics)
    publication_row = _build_publication_html(
        publication_date,
        publication_history,
        show_missing=show_missing_date,
    )
    recommendation_row = _build_recommendation_html(recommendation_matches)
    block_template = """
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%"
      style="width:100%; max-width:900px; margin:0 auto; font-family:Arial, 'Microsoft YaHei', sans-serif; border:1px solid #dddddd; border-radius:8px; background-color:#f9f9f9;">
      <tr>
        <td style="padding:16px 16px 8px; font-size:20px; font-weight:bold; color:#333333; line-height:1.4;">
          {title}
        </td>
      </tr>
      {title_zh_row}
      {source_row}
      {metrics_row}
      <tr>
        <td style="padding:4px 16px; font-size:14px; color:#666666; line-height:1.6;">
          {authors}
        </td>
      </tr>
      {publication_row}
      {recommendation_row}
      <tr>
        <td style="padding:8px 16px 10px;">
          {article_link}
        </td>
      </tr>
      {report_rows}
    </table>
    """
    return block_template.format(
        title=_render_rich_text(title),
        title_zh_row=title_zh_row,
        source_row=source_row,
        metrics_row=metrics_row,
        authors=_render_rich_text(authors),
        publication_row=publication_row,
        recommendation_row=recommendation_row,
        article_link=article_link,
        report_rows=report_rows,
    )


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star" style="display:inline-block; width:0.5em; overflow:hidden; white-space:nowrap;">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ""
    if score >= high:
        return full_star * 5
    interval = (high - low) / 10
    star_num = math.ceil((score - low) / interval)
    full_star_num = int(star_num / 2)
    half_star_num = star_num - full_star_num * 2
    return '<div class="star-wrapper">' + full_star * full_star_num + half_star * half_star_num + "</div>"


def get_summary_html(summary: str) -> str:
    return (
        '<table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" '
        'style="width:100%; max-width:900px; margin:0 auto 16px; font-family:Arial, \'Microsoft YaHei\', sans-serif; border:2px solid #4a90d9; border-radius:10px; background-color:#eef5fb;">'
        '<tr><td style="padding:16px 20px 8px; font-size:18px; font-weight:bold; color:#2c5f8a;">'
        "📋 本期文献速览"
        '</td></tr><tr><td style="padding:0 20px 16px; font-size:14px; color:#333333; line-height:1.7;">'
        + _render_rich_text(summary)
        + "</td></tr></table>"
    )


def _paper_article_url(paper: Paper) -> str:
    for candidate in (getattr(paper, "url", None), getattr(paper, "pdf_url", None)):
        if _safe_url(candidate):
            return candidate
    return ""


def render_email(papers: list[Paper], summary: str | None = None) -> str:
    if not papers:
        return framework.replace("__CONTENT__", get_empty_html())

    parts = []
    if summary:
        parts.append(get_summary_html(summary))

    for paper in papers:
        if isinstance(paper.authors, (list, tuple)):
            author_list = list(paper.authors)
        else:
            author_list = [paper.authors] if _text_value(paper.authors).strip() else []
        authors = author_list[0] if author_list else "作者未提供"

        parts.append(
            get_block_html(
                paper.title,
                authors,
                "",
                getattr(paper, "tldr", None),
                _paper_article_url(paper),
                affiliations=None,
                source=paper.source,
                journal=paper.journal,
                title_zh=paper.title_zh,
                abstract_zh=getattr(paper, "abstract_zh", None),
                reading_card=getattr(paper, "reading_card", None),
                material_scope=getattr(paper, "material_scope", None),
                journal_metrics=getattr(paper, "journal_metrics", None),
                original_abstract=getattr(paper, "abstract", None),
                publication_date=getattr(paper, "publication_date", None),
                publication_history=getattr(paper, "publication_history", None),
                recommendation_matches=getattr(paper, "recommendation_matches", None),
                show_missing_date=True,
            )
        )

    content = '<div style="height:16px; line-height:16px;">&nbsp;</div>'.join(parts)
    return framework.replace("__CONTENT__", content)

