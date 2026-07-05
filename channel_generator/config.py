"""Configuration management for ChannelGenerator."""

from pathlib import Path
from typing import Any, Literal

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
    llm_max_tokens: int | None = Field(default=None, alias="llm_max_tokens")
    llm_thinking_enabled: bool | None = Field(default=None, alias="llm_thinking_enabled")
    llm_reasoning_effort: str | None = Field(default=None, alias="llm_reasoning_effort")
    llm_summary_max_tokens: int | None = Field(default=None, alias="llm_summary_max_tokens")
    llm_summary_thinking_enabled: bool | None = Field(
        default=None,
        alias="llm_summary_thinking_enabled",
    )
    llm_summary_reasoning_effort: str | None = Field(
        default=None,
        alias="llm_summary_reasoning_effort",
    )

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
    concurrency: int = Field(default=8, alias="concurrency")
    dry_run: bool = Field(default=False, alias="dry_run")

    @property
    def effective_summary_model(self) -> str:
        """Return the model used for summary; fallback to main model if not set."""
        return self.llm_summary_model or self.llm_model

    def chat_completion_options(
        self,
        model_role: Literal["main", "summary"] = "main",
    ) -> dict[str, Any]:
        """Return model-specific extra chat completion options.

        The OpenAI-compatible ecosystem is not fully standardized for thinking controls. Keep
        provider-specific fields optional and route them through the SDK's request options.
        """
        thinking_enabled = self.llm_thinking_enabled
        reasoning_effort = self.llm_reasoning_effort
        max_tokens = self.llm_max_tokens

        if model_role == "summary":
            if self.llm_summary_max_tokens is not None:
                max_tokens = self.llm_summary_max_tokens
            if self.llm_summary_thinking_enabled is not None:
                thinking_enabled = self.llm_summary_thinking_enabled
            if self.llm_summary_reasoning_effort:
                reasoning_effort = self.llm_summary_reasoning_effort

        options: dict[str, Any] = {}
        if reasoning_effort:
            options["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            options["max_tokens"] = max_tokens

        extra_body: dict[str, Any] = {}
        if thinking_enabled is not None:
            extra_body["thinking"] = {
                "type": "enabled" if thinking_enabled else "disabled",
            }
        if extra_body:
            options["extra_body"] = extra_body

        return options

    @property
    def manual_keywords(self) -> list[str]:
        """Return manually provided keywords as a list."""
        if not self.search_keywords:
            return []
        return [k.strip() for k in self.search_keywords.split(",") if k.strip()]

    @property
    def effective_concurrency(self) -> int:
        """Return a safe concurrency limit."""
        return max(1, self.concurrency)


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
