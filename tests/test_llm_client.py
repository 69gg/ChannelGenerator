"""Tests for LLM client request options."""

from types import SimpleNamespace
from typing import Any

from channel_generator.config import Settings
from channel_generator.llm_client import LLMClient, tool


class FakeCompletions:
    """Capture chat completion calls."""

    def __init__(self, arguments: list[str] | None = None) -> None:
        self.arguments = arguments or ['{"ok": true}']
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Record kwargs and return a tool-call shaped response."""
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.arguments) - 1)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="hello",
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(arguments=self.arguments[index]),
                            )
                        ],
                    ),
                )
            ],
        )


def _client_with_fake_completions(
    settings: Settings,
    arguments: list[str] | None = None,
) -> tuple[LLMClient, FakeCompletions]:
    client = LLMClient(settings)
    completions = FakeCompletions(arguments)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


async def test_chat_with_tool_includes_main_reasoning_options() -> None:
    """Main-model calls should include configured thinking and effort options."""
    settings = Settings(
        llm_api_key="test",
        llm_max_tokens=81920,
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
    assert completions.calls[0]["max_tokens"] == 81920
    assert "max_completion_tokens" not in completions.calls[0]
    assert completions.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"},
    }


async def test_chat_with_tool_retries_invalid_tool_json() -> None:
    """Malformed tool-call arguments should be retried once."""
    settings = Settings(llm_api_key="test")
    client, completions = _client_with_fake_completions(
        settings,
        arguments=['{"ok": ', '{"ok": true}'],
    )

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
    assert len(completions.calls) == 2


async def test_chat_with_tool_includes_summary_reasoning_options() -> None:
    """Summary-model calls should use summary-specific options when configured."""
    settings = Settings(
        llm_api_key="test",
        llm_model="kimi-k2.6",
        llm_summary_model="deepseek-v4-pro",
        llm_max_tokens=81920,
        llm_thinking_enabled=False,
        llm_reasoning_effort="low",
        llm_summary_max_tokens=81920,
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
    assert completions.calls[0]["max_tokens"] == 81920
    assert "max_completion_tokens" not in completions.calls[0]
    assert completions.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"},
    }
