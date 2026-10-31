"""Tests for the OpenAI-compatible Kimi configuration overrides."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_workflow_llm_overrides_replace_custom_defaults(monkeypatch):
    """The workflow override tuple wins over repository custom.yaml values."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-goaichat-key")
    config_dir = str(Path(__file__).resolve().parent.parent / "config")
    overrides = [
        "llm.api.key=${oc.env:OPENAI_API_KEY}",
        "llm.api.base_url=https://llm.goaichat.top/v1",
        "llm.api.timeout=9000",
        "llm.generation_kwargs.model=kimi-k3",
        "llm.generation_kwargs.max_tokens=65536",
        "llm.max_input_tokens=1000000",
    ]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="default", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()

    assert config.llm.api.key == "fake-goaichat-key"
    assert config.llm.api.base_url == "https://llm.goaichat.top/v1"
    assert config.llm.api.timeout == 9000
    assert config.llm.generation_kwargs.model == "kimi-k3"
    assert config.llm.generation_kwargs.max_tokens == 65536
    assert config.llm.max_input_tokens == 1000000
