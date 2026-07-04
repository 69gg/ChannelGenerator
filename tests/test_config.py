"""Tests for configuration loading."""

from pathlib import Path

from channel_generator.config import load_settings


def test_default_settings():
    """Default settings should be populated."""
    settings = load_settings()
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.keyword_count == 30
    assert settings.max_results == 200
    assert settings.effective_summary_model == "gpt-4o-mini"
    assert settings.manual_keywords == []


def test_manual_keywords_parsing():
    """Manual keywords should be split by comma."""
    settings = load_settings(search_keywords="a, b ,c")
    assert settings.manual_keywords == ["a", "b", "c"]


def test_summary_model_fallback():
    """Summary model falls back to main model when unset."""
    settings = load_settings(llm_model="gpt-4o", llm_summary_model=None)
    assert settings.effective_summary_model == "gpt-4o"
    settings2 = load_settings(llm_model="gpt-4o", llm_summary_model="claude-3-5-sonnet")
    assert settings2.effective_summary_model == "claude-3-5-sonnet"


def test_load_from_toml(tmp_path: Path):
    """Settings should load from a TOML config file."""
    config = tmp_path / "config.toml"
    config.write_text(
        """
llm_base_url = "https://example.com/v1"
llm_api_key = "test-key"
llm_model = "gpt-4o"
keyword_count = 10
max_results = 50
""",
        encoding="utf-8",
    )
    settings = load_settings(config_path=config)
    assert settings.llm_base_url == "https://example.com/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_model == "gpt-4o"
    assert settings.keyword_count == 10
    assert settings.max_results == 50


def test_cli_overrides_toml(tmp_path: Path):
    """CLI-provided overrides should take precedence over TOML."""
    config = tmp_path / "config.toml"
    config.write_text(
        """
[llm]
model = "gpt-4o"
""",
        encoding="utf-8",
    )
    settings = load_settings(config_path=config, llm_model="claude-3-5-sonnet")
    assert settings.llm_model == "claude-3-5-sonnet"
