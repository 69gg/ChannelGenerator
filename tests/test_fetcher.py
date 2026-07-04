"""Tests for the fetcher."""

import pytest
import respx
from httpx import Response

from channel_generator.fetcher import Fetcher


@pytest.fixture
async def fetcher():
    f = Fetcher()
    yield f
    await f.close()


@respx.mock
async def test_fetch_parses_html(fetcher):
    """Fetcher should parse HTML and extract links/title/description."""
    html = """
    <html>
      <head><title>Test Page</title>
      <meta name="description" content="A test page."></head>
      <body>
        <a href="/relative">Relative</a>
        <a href="https://example.com/abs">Absolute</a>
        <a href="#anchor">Anchor</a>
        <p>Hello world</p>
      </body>
    </html>
    """
    route = respx.get("https://example.com/page").mock(return_value=Response(200, text=html))

    snapshot = await fetcher.fetch("https://example.com/page")

    assert route.called
    assert snapshot.status_code == 200
    assert snapshot.title == "Test Page"
    assert snapshot.description == "A test page."
    assert "Hello world" in snapshot.text
    assert "https://example.com/abs" in snapshot.links
    assert any("/relative" in link for link in snapshot.links)
    assert not any(link.startswith("#") for link in snapshot.links)
