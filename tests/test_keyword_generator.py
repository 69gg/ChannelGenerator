"""Tests for keyword generation."""

from unittest.mock import AsyncMock

import pytest

from channel_generator.config import Settings
from channel_generator.keyword_generator import generate_keywords


@pytest.fixture
def settings():
    return Settings(
        llm_api_key="test",
        keyword_count=3,
        search_keywords="seed1, seed2",
    )


async def test_generate_keywords_merges_manual_and_llm(settings):
    """Generated keywords should merge manual seeds with LLM output and dedupe."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={"keywords": ["free ai chat", "Seed2", "new keyword"]}
    )

    keywords = await generate_keywords(client, settings)

    assert "seed1" in [k.lower() for k in keywords]
    assert "seed2" in [k.lower() for k in keywords]
    assert "new keyword" in keywords
    # duplicates removed
    assert len(keywords) == len({k.lower() for k in keywords})
