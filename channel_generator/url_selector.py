"""LLM-driven URL selection from search results and page links using tool calls."""

from dataclasses import dataclass

from channel_generator.llm_client import LLMClient, tool


@dataclass
class CandidateUrl:
    """A candidate URL with metadata."""

    url: str
    title: str
    context: str


SELECT_URLS_TOOL = tool(
    name="select_urls_to_investigate",
    description="Pick the most promising URLs from a Google search result page to investigate further for free LLM/AI chat websites.",
    parameters={
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["url", "reason"],
                },
            },
        },
        "required": ["urls"],
    },
)

EVALUATE_PAGE_TOOL = tool(
    name="evaluate_page_for_channel",
    description="Evaluate whether a fetched page is a free LLM/AI chat channel and select promising outbound links to crawl next.",
    parameters={
        "type": "object",
        "properties": {
            "is_channel": {
                "type": "boolean",
                "description": "Whether the current page itself is a free LLM/AI chat channel.",
            },
            "follow_urls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["url", "reason"],
                },
            },
        },
        "required": ["is_channel", "follow_urls"],
    },
)


def _truncate(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def select_from_search_results(
    client: LLMClient,
    results: list[CandidateUrl],
    max_urls: int,
) -> list[str]:
    """Ask the LLM to pick promising URLs from search results.

    Args:
        client: LLM client.
        results: Search result candidates.
        max_urls: Maximum number of URLs to return.

    Returns:
        Selected URLs.
    """
    if not results:
        return []

    formatted = "\n\n".join(
        f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.context}" for r in results
    )
    user_prompt = (
        f"Pick up to {max_urls} promising URLs from the following search results:\n\n"
        f"{_truncate(formatted)}\n\n"
        f"Return exactly {max_urls} or fewer URLs."
    )
    data = await client.chat_with_tool(
        system_prompt=(
            "You are a web research assistant. Select specific product/site landing pages "
            "that may offer free LLM chat / AI assistant functionality. Exclude news, app stores, "
            "and major platform homepages."
        ),
        user_prompt=user_prompt,
        tool_def=SELECT_URLS_TOOL,
    )
    urls = data.get("urls", [])
    selected: list[str] = []
    for item in urls:
        if isinstance(item, dict) and "url" in item:
            selected.append(item["url"])
    return selected[:max_urls]


async def evaluate_page_and_select_links(
    client: LLMClient,
    url: str,
    title: str,
    description: str,
    page_text: str,
    links: list[str],
    max_follow: int,
) -> tuple[bool, list[str]]:
    """Evaluate a fetched page and optionally select outbound links to follow.

    Args:
        client: LLM client.
        url: Page URL.
        title: Page title.
        description: Page meta description.
        page_text: Visible page text.
        links: Outbound links.
        max_follow: Maximum number of links to follow.

    Returns:
        Tuple of (is_channel, follow_urls).
    """
    links_text = "\n".join(f"- {link}" for link in links[:100])
    user_prompt = (
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Page text:\n{_truncate(page_text)}\n\n"
        f"Outbound links (first 100):\n{links_text}\n\n"
        f"If this page is itself a free LLM/AI chat channel, set is_channel=true. "
        f"Otherwise, pick up to {max_follow} external links that are most likely to lead to such a channel."
    )
    data = await client.chat_with_tool(
        system_prompt=(
            "You are evaluating web pages for a research project. A channel is any site that "
            "offers free LLM/AI chat functionality, including design tools, coding assistants, "
            "writing tools, aggregators, and roleplay platforms."
        ),
        user_prompt=user_prompt,
        tool_def=EVALUATE_PAGE_TOOL,
    )
    is_channel = bool(data.get("is_channel", False))
    follow_items = data.get("follow_urls", [])
    follow_urls: list[str] = []
    for item in follow_items:
        if isinstance(item, dict) and "url" in item:
            follow_urls.append(item["url"])
    return is_channel, follow_urls[:max_follow]
