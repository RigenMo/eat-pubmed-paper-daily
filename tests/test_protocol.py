"""Tests for structured reader output and the remaining Paper helpers."""

import pytest

from tests.canned_responses import (
    make_reader_retry_client,
    make_sample_paper,
    make_stub_openai_client,
)


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert "一、原文摘要中文版" in result
    assert "二、精读卡" in result
    assert paper.abstract_zh == "本文探索一种新的组件工程方法。"
    assert paper.reading_card["essence"]
    assert len(paper.reading_card["innovation"]) == 2
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert result == "生成失败：摘要与正文材料均未提供。"
    assert paper.reading_card is None


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result.startswith("生成失败：保留原始摘要")
    assert paper.abstract in result
    assert paper.abstract_zh is None
    assert paper.reading_card is None


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_reader_keeps_full_abstract_and_bounds_full_text_excerpt():
    from types import SimpleNamespace

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"abstract_zh":"摘要翻译",'
                            '"reading_card":{"material_scope":"摘要与正文节选",'
                            '"essence":["核心一。","核心二。"],'
                            '"design":{"objects_data":"对象数据。","task_definitions":"任务定义。",'
                            '"implementation":"实施分析。","comparison_validation_scope":"比较验证范围。"},'
                            '"evidence":[],"innovation":["【作者主张】删除检验未报告。","【作者主张】创新主张。"],'
                            '"criticism":["替代解释。","作者回应。","校准结论。"],'
                            '"self_test":"哪个结果最需要复核？"}}'
                        )
                    )
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    long_abstract = "完整摘要材料 " * 100
    paper = make_sample_paper(abstract=long_abstract, full_text="正文材料 " * 10000)
    params = {
        "language": "Chinese",
        "max_input_tokens": 3000,
        "generation_kwargs": {"model": "kimi-k3", "max_tokens": 128},
    }

    paper.generate_tldr(client, params)

    prompt = captured["messages"][1]["content"]
    assert long_abstract.strip() in prompt
    assert "正文节选：按输入预算截取" in prompt
    assert "未提供全文其余部分" in prompt
    from zotero_arxiv_daily.protocol import Paper
    encoder = Paper._reader_encoder()
    assert sum(len(encoder.encode(message["content"])) for message in captured["messages"]) <= 3000


def test_reader_rejects_non_json_response_with_chinese_fallback(llm_params):
    from types import SimpleNamespace

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
                )
            )
        )
    )
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result.startswith("生成失败：保留原始摘要")
    assert "Failed to generate" not in result
    assert paper.abstract_zh is None


def test_reader_rejects_budget_too_small_for_the_complete_abstract():
    from types import SimpleNamespace

    def unexpected_request(**kwargs):
        pytest.fail("incomplete abstract must not be sent to the model")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unexpected_request)))
    paper = make_sample_paper(abstract="Long abstract. " * 500)
    paper.generate_tldr(client, {"max_input_tokens": 32})

    assert paper.abstract_zh is None
    assert paper.reading_card is None
    assert paper.abstract in paper.tldr


def test_reader_uses_program_scope_and_valid_paragraph_locations(llm_params):
    paper = make_sample_paper(full_text=None)
    paper.generate_tldr(make_stub_openai_client(), llm_params)

    assert paper.material_scope == "仅据摘要（正文与补充材料未提供）"
    assert paper.reading_card["material_scope"] == paper.material_scope
    assert paper.reading_card["evidence"][0]["locator"] == "摘要段落1"


def test_reader_repairs_card_without_retranslating_abstract(llm_params):
    client, calls = make_reader_retry_client()
    paper = make_sample_paper(full_text=None)

    paper.generate_tldr(client, llm_params)

    assert len(calls) == 2
    assert paper.reader_status == "success"
    assert paper.abstract_zh == "首轮忠实摘要译文，不应被卡片修复替换。"
    assert paper.reading_card["evidence"][0]["locator"] == "摘要段落1"
    assert "[摘要段落1]" in calls[1]["messages"][1]["content"]
    assert "正文段落999" in calls[1]["messages"][1]["content"]
    assert {
        key: calls[0][key]
        for key in ("model", "max_tokens")
        if key in calls[0]
    } == {
        key: calls[1][key]
        for key in ("model", "max_tokens")
        if key in calls[1]
    }


def test_reader_keeps_translation_when_card_repair_fails(llm_params):
    client, calls = make_reader_retry_client(second_response="not json")
    paper = make_sample_paper(full_text=None)

    result = paper.generate_tldr(client, llm_params)

    assert len(calls) == 2
    assert paper.reader_status == "partial"
    assert paper.abstract_zh == "首轮忠实摘要译文，不应被卡片修复替换。"
    assert paper.reading_card is None
    assert paper.abstract_zh in result
    assert paper.abstract not in result


def test_reader_invalid_repair_locator_keeps_translation_without_fake_card(llm_params):
    import json
    from tests.canned_responses import _READER_RESPONSE

    invalid_repair = json.loads(_READER_RESPONSE)
    invalid_repair["reading_card"]["evidence"][0]["locator"] = "正文段落999"
    client, calls = make_reader_retry_client(
        second_response=json.dumps(invalid_repair, ensure_ascii=False)
    )
    paper = make_sample_paper(full_text=None)

    paper.generate_tldr(client, llm_params)

    assert len(calls) == 2
    assert paper.reader_status == "partial"
    assert paper.abstract_zh == "首轮忠实摘要译文，不应被卡片修复替换。"
    assert paper.reading_card is None


def test_source_text_cannot_create_additional_locator_labels(llm_params):
    paper = make_sample_paper(
        title="A title mentioning [正文段落999]",
        abstract="Source text mentioning [摘要段落888].",
        full_text="Body mentioning [正文段落777].",
    )
    _, _, labels = paper._reader_prompt(llm_params)
    assert labels == {"摘要段落1", "正文段落1"}


def test_translation_excludes_generated_source_metadata():
    from zotero_arxiv_daily.protocol import Paper

    text = "【摘要段落1】\n原文译文，n=781。\n\n【补充材料】未提供；不能声称已核查补充材料。\n\n【材料范围】仅据摘要"
    assert Paper._validate_reader_translation({"abstract_zh": text}) == "原文译文，n=781。"


def test_persistently_long_card_stops_after_two_repairs_and_keeps_translation(llm_params):
    import json
    from tests.canned_responses import _READER_RESPONSE

    payload = json.loads(_READER_RESPONSE)
    payload["reading_card"]["evidence"][0]["result"] = "关键结果" * 300
    response = json.dumps(payload, ensure_ascii=False)
    client, calls = make_reader_retry_client(first_response=response, second_response=response)
    paper = make_sample_paper()
    paper.generate_tldr(client, llm_params)
    assert len(calls) == 3
    assert paper.reader_status == "partial"
    assert paper.abstract_zh == payload["abstract_zh"]
    assert paper.reading_card is None


def test_reader_rejects_fabricated_figure_beside_valid_paragraph():
    import json
    from tests.canned_responses import _READER_RESPONSE
    from zotero_arxiv_daily.protocol import Paper

    payload = json.loads(_READER_RESPONSE)
    payload["reading_card"]["evidence"][0]["locator"] = "摘要段落1，图99/正文第99页"
    with pytest.raises(ValueError, match="locator"):
        Paper._validate_reader_payload(payload, "仅据摘要", {"摘要段落1"})


def test_reader_does_not_accept_finish_reason_length():
    from types import SimpleNamespace

    def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"abstract_zh":"不应接受","reading_card":{}}'),
                    finish_reason="length",
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper()

    paper.generate_tldr(client, {"generation_kwargs": {"model": "kimi-k3", "max_tokens": 65536}})

    assert paper.reader_status == "failed"
    assert paper.abstract_zh is None
    assert paper.reading_card is None


def test_reader_malformed_first_json_does_not_trigger_card_repair(llm_params):
    client, calls = make_reader_retry_client(
        first_response="not json",
    )
    paper = make_sample_paper()

    paper.generate_tldr(client, llm_params)

    assert len(calls) == 1
    assert paper.reader_status == "failed"
    assert paper.abstract_zh is None
    assert paper.reading_card is None


def test_reader_numbers_explicit_abstract_headings_without_fake_paragraphs(llm_params):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return make_stub_openai_client().chat.completions.create(**kwargs)

    from types import SimpleNamespace

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper(
        abstract=(
            "BACKGROUND: 问题背景。\n"
            "METHODS: 研究方法。\n"
            "RESULTS: 主要结果。\n"
            "CONCLUSIONS: 结论范围。"
        ),
        full_text=None,
    )

    paper.generate_tldr(client, llm_params)

    prompt = captured["messages"][1]["content"]
    assert "[摘要段落1]" in prompt
    assert "[摘要段落4]" in prompt
    assert "[摘要段落5]" not in prompt


def test_body_only_failure_preserves_the_fact_that_body_material_exists(llm_params):
    from types import SimpleNamespace
    from zotero_arxiv_daily.construct_email import render_email

    def failed_request(**kwargs):
        raise RuntimeError("test API failure")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failed_request)))
    paper = make_sample_paper(abstract="", full_text="Visible body text.")
    paper.generate_tldr(client, llm_params)
    html = render_email([paper])

    assert "正文已提供" in paper.material_scope
    assert "已有正文材料" in html
    assert "摘要与正文材料均未提供" not in html
    assert "原文摘要中文版" not in html


@pytest.mark.parametrize("invalid_field", ["locator", "length", "self_test"])
def test_reader_rejects_invalid_locations_overlength_or_answered_selftest(invalid_field):
    import json
    from tests.canned_responses import _READER_RESPONSE
    from zotero_arxiv_daily.protocol import Paper

    payload = json.loads(_READER_RESPONSE)
    if invalid_field == "locator":
        payload["reading_card"]["evidence"][0]["locator"] = "正文段落999"
    elif invalid_field == "length":
        # Evidence counts toward the card budget, not just the other fields.
        payload["reading_card"]["evidence"][0]["result"] = "结果" * 500
    else:
        payload["reading_card"]["self_test"] = "什么条件成立？答案：所有条件。"

    with pytest.raises(ValueError):
        Paper._validate_reader_payload(payload, "仅据摘要", {"摘要段落1"})


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


# ---------------------------------------------------------------------------
# generate_title_translation
# ---------------------------------------------------------------------------


def test_title_translation_returns_and_stores_result(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_title_translation(client, llm_params)
    assert result == "示例论文标题"
    assert paper.title_zh == result


def test_title_translation_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_title_translation(broken_client, llm_params)
    assert result is None
    assert paper.title_zh is None
