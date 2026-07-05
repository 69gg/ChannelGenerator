"""Tests for LLM client request options."""

from types import SimpleNamespace
from typing import Any

from channel_generator.config import Settings
from channel_generator.llm_client import LLMClient, tool


class FakeCompletions:
    """Capture chat completion calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Record kwargs and return a tool-call shaped response."""
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="hello",
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(arguments='{"ok": true}'),
                            )
                        ],
                    ),
                )
            ],
        )


def _client_with_fake_completions(settings: Settings) -> tuple[LLMClient, FakeCompletions]:
    client = LLMClient(settings)
    completions = FakeCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


async def test_chat_with_tool_includes_main_reasoning_options() -> None:
    """Main-model calls should include configured thinking and effort options."""
    settings = Settings(
        llm_api_key="test",
        llm_thinking_enabled=True,
        llm_reasoning_effort="high",
    )
    client, completions = _client_with_fake_completions(settings)

    result = await client.chat_with_tool(
        system_prompt="system",
        user_prompt="user",
        tool_def=tool(
            name="record",
            description="Record a result.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    )

    assert result == {"ok": True}
    assert completions.calls[0]["reasoning_effort"] == "high"
    assert completions.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"},
    }


async def test_chat_with_tool_includes_summary_reasoning_options() -> None:
    """Summary-model calls should use summary-specific options when configured."""
    settings = Settings(
        llm_api_key="test",
        llm_model="kimi-k2.6",
        llm_summary_model="deepseek-v4-pro",
        llm_thinking_enabled=False,
        llm_reasoning_effort="low",
        llm_summary_thinking_enabled=True,
        llm_summary_reasoning_effort="high",
    )
    client, completions = _client_with_fake_completions(settings)

    await client.chat_with_tool(
        system_prompt="system",
        user_prompt="user",
        tool_def=tool(
            name="record",
            description="Record a result.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        model=settings.effective_summary_model,
        model_role="summary",
    )

    assert completions.calls[0]["model"] == "deepseek-v4-pro"
    assert completions.calls[0]["reasoning_effort"] == "high"
    assert completions.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"},
    }
