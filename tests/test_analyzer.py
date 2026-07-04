"""Tests for the analyzer."""

from unittest.mock import AsyncMock

from channel_generator.analyzer import analyze_page
from channel_generator.fetcher import PageSnapshot


async def test_analyze_page_returns_channel_info():
    """Analyzer should parse LLM tool-call output into ChannelInfo."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={
            "is_free_llm_chat": True,
            "name": "ExampleChat",
            "description": "Free AI chat",
            "models": ["GPT-4o"],
            "free_tier_desc": "Daily 20 messages",
            "requires_auth": True,
            "category": "pure_chat",
            "confidence": "high",
            "notes": "",
        }
    )

    snapshot = PageSnapshot(
        url="https://example.com/chat",
        status_code=200,
        title="Example Chat",
        description="Chat with AI",
        text="Free AI chat daily",
        links=[],
        html="",
    )
    info = await analyze_page(client, snapshot)

    assert info is not None
    assert info.name == "ExampleChat"
    assert info.category == "pure_chat"
    assert info.confidence == "high"


async def test_analyze_page_returns_none_for_irrelevant():
    """Analyzer should return None when page is not a channel."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(return_value={"is_free_llm_chat": False})

    snapshot = PageSnapshot(
        url="https://example.com/news",
        status_code=200,
        title="News",
        description="",
        text="Some news",
        links=[],
        html="",
    )
    info = await analyze_page(client, snapshot)

    assert info is None
