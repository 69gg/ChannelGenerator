"""Tests for Google search source."""

from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response

from channel_generator.fetcher import Fetcher
from channel_generator.sources.google_search import GoogleSearchSource


@pytest.fixture
async def fetcher():
    f = Fetcher()
    yield f
    await f.close()


@respx.mock
async def test_google_search_parses_results(fetcher):
    """Google search source should extract results via LLM tool call."""
    html = """
    <html><body>
      <a href="https://example.com/chat"><h3>Example Chat</h3></a>
      <div>A free AI chat website.</div>
      <a href="/internal"><h3>Internal</h3></a>
    </body></html>
    """
    route = respx.get("https://www.google.com/search?q=free+AI+chat").mock(
        return_value=Response(200, text=html)
    )

    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Example Chat",
                    "url": "https://example.com/chat",
                    "snippet": "A free AI chat website.",
                },
            ]
        }
    )

    source = GoogleSearchSource(fetcher, client)
    results = await source.search("free AI chat")

    assert route.called
    assert len(results) == 1
    assert results[0].url == "https://example.com/chat"
    assert results[0].title == "Example Chat"
    assert "free AI chat" in results[0].snippet
    client.chat_with_tool.assert_awaited_once()
