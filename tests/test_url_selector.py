"""Tests for URL selector."""

from unittest.mock import AsyncMock

from channel_generator.url_selector import (
    CandidateUrl,
    evaluate_page_and_select_links,
    select_from_search_results,
)


async def test_select_from_search_results():
    """Selector should return URLs from LLM response."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={
            "urls": [
                {"url": "https://a.com", "reason": "ok"},
                {"url": "https://b.com", "reason": "ok"},
            ]
        }
    )

    candidates = [
        CandidateUrl("A", "https://a.com", "free chat"),
        CandidateUrl("B", "https://b.com", "AI assistant"),
    ]
    urls = await select_from_search_results(client, candidates, max_urls=2)

    assert urls == ["https://a.com", "https://b.com"]


async def test_evaluate_page_and_select_links():
    """Page evaluator should parse is_channel and follow_urls."""
    client = AsyncMock()
    client.chat_with_tool = AsyncMock(
        return_value={
            "is_channel": False,
            "follow_urls": [
                {"url": "https://next.com", "reason": "looks promising"},
            ],
        }
    )

    is_channel, follow = await evaluate_page_and_select_links(
        client,
        url="https://current.com",
        title="Current",
        description="desc",
        page_text="some text",
        links=["https://next.com", "https://other.com"],
        max_follow=1,
    )

    assert is_channel is False
    assert follow == ["https://next.com"]
