"""Configuration management for ChannelGenerator."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or config.toml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # LLM settings
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="llm_base_url")
    llm_api_key: str = Field(default="", alias="llm_api_key")
    llm_model: str = Field(default="gpt-5.5", alias="llm_model")
    llm_summary_model: str | None = Field(default=None, alias="llm_summary_model")

    # Search / crawl settings
    keyword_count: int = Field(default=30, alias="keyword_count")
    keyword_agents: int = Field(default=8, alias="keyword_agents")
    urls_per_search_page: int = Field(default=5, alias="urls_per_search_page")
    recursion_depth: int = Field(default=2, alias="recursion_depth")
    urls_per_recursion: int = Field(default=3, alias="urls_per_recursion")
    max_search_per_query: int = Field(default=20, alias="max_search_per_query")

    # Optional manual keywords, comma separated
    search_keywords: str | None = Field(default=None, alias="search_keywords")

    # Output settings
    max_results: int = Field(default=200, alias="max_results")
    report_path: Path = Field(default=Path("report.md"), alias="report_path")
    state_path: Path = Field(default=Path("state.json"), alias="state_path")

    # Runtime
    dry_run: bool = Field(default=False, alias="dry_run")

    @property
    def effective_summary_model(self) -> str:
        """Return the model used for summary; fallback to main model if not set."""
        return self.llm_summary_model or self.llm_model

    @property
    def manual_keywords(self) -> list[str]:
        """Return manually provided keywords as a list."""
        if not self.search_keywords:
            return []
        return [k.strip() for k in self.search_keywords.split(",") if k.strip()]


def load_settings(
    config_path: Path | None = None,
    **overrides,
) -> Settings:
    """Load settings from optional config.toml and environment variables.

    Args:
        config_path: Optional path to a TOML config file.
        **overrides: Additional keyword overrides.

    Returns:
        Populated Settings instance.
    """
    kwargs = dict(overrides)
    if config_path and config_path.exists():
        import tomllib

        with config_path.open("rb") as f:
            data = tomllib.load(f)
        # Support both flat and nested TOML: flatten nested sections into kwargs.
        for section_name, section_values in data.items():
            if isinstance(section_values, dict):
                prefix = f"{section_name}_"
                for key, value in section_values.items():
                    full_key = key if key.startswith(prefix) else f"{prefix}{key}"
                    kwargs.setdefault(full_key, value)
            else:
                kwargs.setdefault(section_name, section_values)
    return Settings(**kwargs)
