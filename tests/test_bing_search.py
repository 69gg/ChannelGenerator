"""Tests for Bing search source."""

from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response

from channel_generator.fetcher import Fetcher
from channel_generator.sources.bing_search import BingSearchSource


@pytest.fixture
async def fetcher():
    f = Fetcher()
    yield f
    await f.close()


@respx.mock
async def test_bing_search_parses_results(fetcher):
    """Bing search source should extract results via LLM tool call."""
    html = """
    <html><body>
      <a href="https://example.com/chat"><h3>Example Chat</h3></a>
      <p>Free AI chat website.</p>
    </body></html>
    """
    route = respx.get("https://www.bing.com/search?q=free+AI+chat").mock(
        return_value=Response(200, text=html)
    )

    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Example Chat",
                    "url": "https://example.com/chat",
                    "snippet": "Free AI chat website.",
                },
            ]
        }
    )

    source = BingSearchSource(fetcher, client)
    results = await source.search("free AI chat")

    assert route.called
    assert len(results) == 1
    assert results[0].url == "https://example.com/chat"
    client.chat_with_tool.assert_awaited_once()
