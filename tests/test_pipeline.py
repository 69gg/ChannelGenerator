"""End-to-end pipeline smoke test with mocked LLM and HTTP."""

from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response

from channel_generator.config import Settings
from channel_generator.pipeline import DiscoveryPipeline


@pytest.fixture
def settings(tmp_path):
    return Settings(
        llm_api_key="test-key",
        llm_model="gpt-4o-mini",
        llm_summary_model="gpt-4o-mini",
        keyword_count=1,
        keyword_agents=2,
        urls_per_search_page=1,
        recursion_depth=1,
        urls_per_recursion=1,
        max_results=10,
        report_path=tmp_path / "report.md",
        state_path=tmp_path / "state.json",
    )


@respx.mock
async def test_pipeline_end_to_end(settings):
    """Pipeline should generate keywords, crawl, analyze, and write report."""
    # Google search result page
    google_html = """
    <html><body>
      <a href="https://example.com/chat"><h3>Example Chat</h3></a>
      <div>Free AI chat website.</div>
    </body></html>
    """
    google_route = respx.get("https://www.google.com/search?q=free+ai+chat").mock(
        return_value=Response(200, text=google_html)
    )

    # Bing search result page
    bing_html = """
    <html><body>
      <a href="https://example.com/chat"><h3>Example Chat on Bing</h3></a>
      <p>Free AI chat.</p>
    </body></html>
    """
    bing_route = respx.get("https://www.bing.com/search?q=free+ai+chat").mock(
        return_value=Response(200, text=bing_html)
    )

    # Target page
    target_html = """
    <html><head><title>Example Chat - Free AI</title></head>
    <body>
      <h1>Chat with GPT-4o for free</h1>
      <p>Free daily quota, no credit card required.</p>
      <a href="https://other.com">Other</a>
    </body></html>
    """
    target_route = respx.get("https://example.com/chat").mock(
        return_value=Response(200, text=target_html)
    )

    pipeline = DiscoveryPipeline(settings)

    # Mock LLM responses in expected order:
    # 1. keyword generation (agent 1)
    # 2. keyword generation (agent 2)
    # 3. google search result extraction
    # 4. bing search result extraction
    # 5. url selection from merged search results
    # 6. page evaluation (is_channel + follow links)
    # 7. channel analysis
    # 8. summarization
    pipeline.client.chat_with_tool = AsyncMock(
        side_effect=[
            {"keywords": ["free ai chat"]},
            {"keywords": ["ai driven design free"]},
            {"results": [{"title": "Example Chat", "url": "https://example.com/chat", "snippet": "Free AI chat"}]},
            {"results": [{"title": "Example Chat on Bing", "url": "https://example.com/chat", "snippet": "Free AI chat"}]},
            {"urls": [{"url": "https://example.com/chat", "reason": "free chat"}]},
            {"is_channel": True, "follow_urls": []},
            {
                "is_free_llm_chat": True,
                "name": "ExampleChat",
                "description": "Free AI chat",
                "models": ["GPT-4o"],
                "free_tier_desc": "Daily free quota",
                "requires_auth": False,
                "category": "pure_chat",
                "confidence": "high",
                "notes": "",
            },
            {
                "summary": "Found one channel.",
                "highlights": ["ExampleChat offers free GPT-4o chat"],
                "category_counts": {"pure_chat": 1},
                "confidence_counts": {"high": 1, "medium": 0, "low": 0},
            },
        ]
    )

    try:
        channels = await pipeline.run()
    finally:
        await pipeline.close()

    assert google_route.called
    assert bing_route.called
    assert target_route.called
    assert len(channels) == 1
    assert channels[0].name == "ExampleChat"
    assert settings.report_path.exists()
    report_text = settings.report_path.read_text(encoding="utf-8")
    assert "ExampleChat" in report_text
    assert settings.state_path.exists()
