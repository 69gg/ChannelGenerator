"""Tests for the summarizer."""

from unittest.mock import AsyncMock

from channel_generator.config import Settings
from channel_generator.summarizer import summarize_channels


async def test_summarize_channels_empty():
    """Empty channel list should return a default summary."""
    client = AsyncMock()
    settings = Settings(llm_api_key="test")
    result = await summarize_channels(client, settings, [])

    assert result["summary"] == "No channels discovered in this run."
    assert result["confidence_counts"] == {"high": 0, "medium": 0, "low": 0}


async def test_summarize_channels_uses_tool():
    """Summarizer should call the LLM with a tool."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={
            "summary": "Found channels.",
            "highlights": ["A"],
            "category_counts": {"pure_chat": 2},
            "confidence_counts": {"high": 2, "medium": 0, "low": 0},
        }
    )
    settings = Settings(llm_api_key="test", llm_summary_model="gpt-4o")
    channels = [
        {"name": "A", "category": "pure_chat", "confidence": "high"},
        {"name": "B", "category": "pure_chat", "confidence": "high"},
    ]
    result = await summarize_channels(client, settings, channels)

    assert result["summary"] == "Found channels."
    assert result["category_counts"] == {"pure_chat": 2}
    client.chat_with_tool.assert_awaited_once()
