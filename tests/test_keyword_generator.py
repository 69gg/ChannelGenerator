"""Tests for keyword generation."""

from unittest.mock import AsyncMock

import pytest

from channel_generator.config import Settings
from channel_generator.keyword_generator import PERSPECTIVES, generate_keywords


@pytest.fixture
def settings():
    return Settings(
        llm_api_key="test",
        keyword_count=6,
        keyword_agents=2,
        search_keywords="seed1, seed2",
    )


async def test_generate_keywords_uses_multiple_agents(settings):
    """Generated keywords should merge outputs from multiple parallel agents."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        side_effect=[
            {"keywords": ["free ai chat"]},
            {"keywords": ["ai driven design free"]},
        ]
    )

    keywords = await generate_keywords(client, settings)

    assert client.chat_with_tool.await_count == 2
    assert "free ai chat" in [k.lower() for k in keywords]
    assert "ai driven design free" in [k.lower() for k in keywords]
    assert "seed1" in [k.lower() for k in keywords]


async def test_generate_keywords_merges_manual_and_dedupes(settings):
    """Manual seeds should be merged and duplicates removed."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        side_effect=[
            {"keywords": ["seed2", "free coding assistant"]},
            {"keywords": ["Seed2", "ai writer free"]},
        ]
    )

    keywords = await generate_keywords(client, settings)

    # manual seeds come first, then LLM outputs deduped
    lower = [k.lower() for k in keywords]
    assert lower.count("seed2") == 1
    assert "seed1" in lower
    assert "free coding assistant" in lower
    assert "ai writer free" in lower


def test_perspectives_cover_broad_areas():
    """There should be at least 8 keyword perspectives covering broad areas."""
    assert len(PERSPECTIVES) >= 8
    names = {p["name"] for p in PERSPECTIVES}
    assert "ai_driven_apps" in names
    assert "coding_assistants" in names
    assert "writing_office" in names
