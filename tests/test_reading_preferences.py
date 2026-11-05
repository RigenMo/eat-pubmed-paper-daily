from datetime import date

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.reading_preferences import effective_focus_mode, focus_queries, focus_evidence, order_candidates


def test_temporary_focus_can_be_disabled_or_expire():
    settings = {"focus_mode": "ukb_multimorbidity", "focus_until": "2026-11-20"}
    assert effective_focus_mode(settings, date(2026, 11, 20)) == "ukb_multimorbidity"
    assert effective_focus_mode(settings, date(2026, 11, 21)) == "balanced"
    assert effective_focus_mode({"focus_mode": "balanced"}) == "balanced"
    with pytest.raises(ValueError):
        effective_focus_mode({"focus_mode": "unknown"})


def test_focus_is_explicit_interest_and_not_a_fake_zotero_reference():
    paper = make_sample_paper(title="A new dimension of multimorbidity in UK Biobank")
    evidence = focus_evidence(paper, "ukb_multimorbidity")
    assert evidence["basis"] == "current_focus"
    assert evidence["candidate_quote"] in paper.title
    assert "anchor_title" not in evidence and "anchor_key" not in evidence
    assert focus_evidence(paper, "balanced") is None
    assert focus_queries("balanced") == []
    assert len(focus_queries("ukb_multimorbidity")) == 3


def test_cognitive_value_comes_before_resource_scale_or_recency():
    newer = make_sample_paper(title="UK Biobank enlarged resource", publication_date="2026-11-05")
    older = make_sample_paper(title="Reframing a disease question", publication_date="2026-10-30")
    newer.innovation_assessment = {"type": "capability"}
    older.innovation_assessment = {"type": "cognitive"}
    assert order_candidates([newer, older], "ukb_multimorbidity") == [older, newer]


def test_within_innovation_type_focus_and_then_recency_apply():
    focus = make_sample_paper(title="UK Biobank disease trajectories", publication_date="2026-10-30")
    other = make_sample_paper(title="Another cognitive contribution", publication_date="2026-11-05")
    for paper in (focus, other):
        paper.innovation_assessment = {"type": "cognitive"}
    assert order_candidates([other, focus], "ukb_multimorbidity") == [focus, other]
    assert order_candidates([other, focus], "balanced") == [other, focus]


def test_workflow_overrides_legacy_five_paper_settings(monkeypatch):
    import shlex
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("RECOMMENDATION_FOCUS", "balanced")
    root = Path(__file__).resolve().parents[1]
    line = next(line.strip() for line in (root / '.github/workflows/main.yml').read_text(encoding='utf-8').splitlines() if 'uv run src/' in line)
    overrides = ['executor.max_paper_num=5', 'executor.debug=true', 'recommendations.max_papers=5'] + shlex.split(line)[3:]
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(root / 'config')):
            config = compose(config_name='default', overrides=overrides)
        assert config.executor.max_paper_num == config.recommendations.max_papers == 10
        assert config.executor.debug is False
        assert config.search.mode == 5 and config.source.mix_mode == 1
        assert effective_focus_mode(config.recommendations) == 'balanced'
    finally:
        GlobalHydra.instance().clear()
