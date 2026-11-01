"""Shared stub factories for tests. No unittest.mock anywhere."""

from datetime import datetime
from types import SimpleNamespace
import json

from zotero_arxiv_daily.protocol import CorpusPaper, Paper


# ---------------------------------------------------------------------------
# OpenAI client stub
# ---------------------------------------------------------------------------

_AFFILIATION_MARKER = "You are an assistant who perfectly extracts affiliations"
_AFFILIATION_RESPONSE = '["TsingHua University","Peking University"]'
_TITLE_TRANSLATION_MARKER = "professional academic translator"
_TITLE_TRANSLATION_RESPONSE = "示例论文标题"
_TLDR_RESPONSE = "Hello! How can I assist you today?"
_READER_MARKER = "结构化中文文献阅读卡"
_READER_RESPONSE = json.dumps(
    {
        "abstract_zh": "本文探索一种新的组件工程方法。",
        "reading_card": {
            "material_scope": "摘要与正文",
            "essence": ["研究提出一个新的组件工程方法。", "结果显示该方法在给定材料中具有可观察的改进。"],
            "design": {
                "objects_data": "对象与数据：示例组件工程数据。",
                "task_definitions": "研究任务与关键定义：比较新方法和基线方法。",
                "implementation": "实施/分析：按论文所述流程完成分析。",
                "comparison_validation_scope": "比较、验证与适用范围：结果仅适用于所报告的研究场景。",
            },
            "evidence": [
                {
                    "claim": "新方法表现更好",
                    "comparison": "相对于基线方法",
                    "result": "报告了改进结果",
                    "locator": "摘要段落1",
                }
            ],
            "innovation": [
                "【推断】去掉该方法新增的组件，材料中的核心比较便无法回答。",
                "【作者主张】该方法为组件工程提供了新的实现路径。",
            ],
            "criticism": [
                "替代解释：差异也可能来自样本或实施条件。",
                "作者回应：材料中仅报告了既定比较，未充分排除该解释。",
                "校准结论：结果支持相关改进，但不能单独证明普遍因果效果。",
            ],
            "self_test": "如果去掉关键组件，结果是否仍然成立？",
        },
    },
    ensure_ascii=False,
)


def _reader_response_with_invalid_locator() -> str:
    payload = json.loads(_READER_RESPONSE)
    payload["abstract_zh"] = "首轮忠实摘要译文，不应被卡片修复替换。"
    payload["reading_card"]["evidence"][0]["locator"] = "正文段落999"
    return json.dumps(payload, ensure_ascii=False)


def _reader_repair_response() -> str:
    payload = json.loads(_READER_RESPONSE)
    payload["abstract_zh"] = "第二次不应替换首轮摘要译文。"
    return json.dumps(payload, ensure_ascii=False)


def _make_chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
                index=0,
            )
        ],
        id="chatcmpl-stub",
        created=1765197615,
        model="gpt-4o-mini-2024-07-18",
        object="chat.completion",
    )


def _stub_chat_create(**kwargs):
    messages = kwargs.get("messages", [])
    request_str = str(messages)
    if _AFFILIATION_MARKER in request_str:
        return _make_chat_response(_AFFILIATION_RESPONSE)
    if _TITLE_TRANSLATION_MARKER in request_str:
        return _make_chat_response(_TITLE_TRANSLATION_RESPONSE)
    if _READER_MARKER in request_str:
        return _make_chat_response(_READER_RESPONSE)
    return _make_chat_response(_TLDR_RESPONSE)


def _stub_embeddings_create(**kwargs):
    inputs = kwargs.get("input", [])
    n = len(inputs) if isinstance(inputs, list) else 1
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=i, object="embedding") for i in range(n)],
        model="text-embedding-3-large",
        object="list",
    )


def make_stub_openai_client():
    """Return a SimpleNamespace that quacks like openai.OpenAI().

    chat.completions.create() and embeddings.create() behave identically
    to the Docker mock_openai server that CI previously relied on.
    """
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_stub_chat_create),
        ),
        embeddings=SimpleNamespace(create=_stub_embeddings_create),
    )


def make_reader_retry_client(first_response=None, second_response=None):
    """Return a deterministic reader client and the captured reader calls."""
    calls = []
    responses = [
        first_response or _reader_response_with_invalid_locator(),
        second_response or _reader_repair_response(),
    ]

    def create(**kwargs):
        request_str = str(kwargs.get("messages", []))
        if _READER_MARKER in request_str:
            calls.append(kwargs)
            index = min(len(calls) - 1, len(responses) - 1)
            return _make_chat_response(responses[index])
        return _stub_chat_create(**kwargs)

    return (
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            embeddings=SimpleNamespace(create=_stub_embeddings_create),
        ),
        calls,
    )


# ---------------------------------------------------------------------------
# Zotero client stub
# ---------------------------------------------------------------------------

_DEFAULT_COLLECTIONS = [
    {
        "key": "COL1",
        "data": {"name": "survey", "parentCollection": False},
    },
    {
        "key": "COL2",
        "data": {"name": "topic-a", "parentCollection": "COL1"},
    },
]

_DEFAULT_ITEMS = [
    {
        "data": {
            "title": "Stub Paper 1",
            "abstractNote": "Abstract of stub paper 1.",
            "dateAdded": "2026-01-15T10:00:00Z",
            "collections": ["COL2"],
        },
    },
    {
        "data": {
            "title": "Stub Paper 2",
            "abstractNote": "Abstract of stub paper 2.",
            "dateAdded": "2026-02-20T12:00:00Z",
            "collections": ["COL1"],
        },
    },
]


def make_stub_zotero_client(collections=None, items=None):
    """Return a SimpleNamespace that quacks like pyzotero.zotero.Zotero.

    Supports the call patterns used by Executor.fetch_zotero_corpus():
        zot.everything(zot.collections())
        zot.everything(zot.items(itemType=...))
    """
    cols = collections if collections is not None else _DEFAULT_COLLECTIONS
    itms = items if items is not None else _DEFAULT_ITEMS

    def everything(generator):
        return generator

    def collections_fn():
        return cols

    def items_fn(**kwargs):
        return itms

    return SimpleNamespace(
        everything=everything,
        collections=collections_fn,
        items=items_fn,
    )


# ---------------------------------------------------------------------------
# SMTP stub
# ---------------------------------------------------------------------------


def make_stub_smtp(sent_emails: list):
    """Return a class that records calls to sendmail().

    Usage:
        sent = []
        monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
        ...
        assert len(sent) == 1
        sender, recipients, body = sent[0]
    """

    class StubSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, sender, recipients, msg):
            sent_emails.append((sender, recipients, msg))

        def quit(self):
            pass

    return StubSMTP


# ---------------------------------------------------------------------------
# Paper / CorpusPaper factories
# ---------------------------------------------------------------------------


def make_sample_paper(**overrides) -> Paper:
    defaults = dict(
        source="arxiv",
        title="Sample Paper Title",
        authors=["Author A", "Author B", "Author C"],
        abstract="This paper explores a novel approach to widget engineering.",
        url="https://arxiv.org/abs/2026.00001",
        pdf_url="https://arxiv.org/pdf/2026.00001",
        full_text="\\begin{document} Some text. \\end{document}",
        tldr=None,
        affiliations=None,
        score=None,
    )
    defaults.update(overrides)
    return Paper(**defaults)


def make_sample_corpus(n: int = 3) -> list[CorpusPaper]:
    return [
        CorpusPaper(
            title=f"Corpus Paper {i}",
            abstract=f"Abstract for corpus paper {i}.",
            added_date=datetime(2026, 1, 1 + i),
            paths=[f"2026/survey/topic-{i}"],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# bioRxiv canned API response
# ---------------------------------------------------------------------------

SAMPLE_BIORXIV_API_RESPONSE = {
    "messages": [{"status": "ok"}],
    "collection": [
        {
            "doi": "10.1101/2026.03.01.000001",
            "title": "A biorxiv paper",
            "authors": "Smith, J.; Doe, A.; Lee, K.",
            "abstract": "We present a novel finding.",
            "date": "2026-03-02",
            "category": "bioinformatics",
            "version": "1",
        },
        {
            "doi": "10.1101/2026.03.01.000002",
            "title": "Another biorxiv paper",
            "authors": "Wang, L.; Chen, M.",
            "abstract": "We replicate a key result.",
            "date": "2026-03-02",
            "category": "genomics",
            "version": "1",
        },
        {
            "doi": "10.1101/2026.03.01.000003",
            "title": "Old biorxiv paper",
            "authors": "Old, R.",
            "abstract": "Yesterday's paper.",
            "date": "2026-03-01",
            "category": "bioinformatics",
            "version": "1",
        },
    ],
}


# ---------------------------------------------------------------------------
# PubMed (NCBI E-utilities) canned responses
# ---------------------------------------------------------------------------

SAMPLE_PUBMED_ESEARCH_RESPONSE = {
    "esearchresult": {"idlist": ["30000001", "30000002"]},
}

# MEDLINE flat-file text as returned by efetch (rettype=medline, retmode=text).
# Field layout: 4-char tag, "- ", value (see PubMedRetriever._parse_medline).
SAMPLE_PUBMED_MEDLINE_TEXT = (
    "PMID- 30000001\n"
    "TI  - A sample PubMed paper on AI diagnostics\n"
    "AB  - This is a sample abstract about AI diagnostics in ophthalmology.\n"
    "AU  - Smith J\n"
    "AU  - Doe A\n"
    "AD  - Department of Ophthalmology, Harvard Medical School, Boston, MA 02115, USA.\n"
    "AD  - Department of Ophthalmology, Harvard Medical School, Boston, MA 02115, USA.\n"
    "JT  - Nature Medicine\n"
    "LID - 10.1038/s41591-026-00001-1 [doi]\n"
    "\n"
    "PMID- 30000002\n"
    "TI  - Another PubMed paper without affiliation\n"
    "AB  - Second sample abstract text.\n"
    "AU  - Lee K\n"
    "JT  - The Lancet\n"
    "\n"
)
