"""Direct Google search result page scraping via LLM tool-call extraction."""

from dataclasses import dataclass
from urllib.parse import quote_plus

from channel_generator.fetcher import Fetcher
from channel_generator.llm_client import LLMClient, tool


@dataclass
class GoogleResult:
    """A single Google search result."""

    title: str
    url: str
    snippet: str


SYSTEM_PROMPT = """You are parsing a Google search result page HTML.

Extract all organic search results. Exclude ads, navigation links, and internal Google links.
"""

EXTRACT_RESULTS_TOOL = tool(
    name="extract_search_results",
    description="Extract organic search results from a Google result page HTML.",
    parameters={
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["title", "url", "snippet"],
                },
            },
        },
        "required": ["results"],
    },
)


class GoogleSearchSource:
    """Fetch Google result pages and use LLM tool calls to extract results."""

    def __init__(self, fetcher: Fetcher, client: LLMClient | None = None) -> None:
        self.fetcher = fetcher
        self.client = client

    def build_url(self, query: str, start: int = 0) -> str:
        """Build a Google search URL."""
        encoded = quote_plus(query)
        if start:
            return f"https://www.google.com/search?q={encoded}&start={start}"
        return f"https://www.google.com/search?q={encoded}"

    async def search(self, query: str, start: int = 0) -> list[GoogleResult]:
        """Fetch a Google result page and extract results via LLM tool call.

        Args:
            query: Search query.
            start: Result offset.

        Returns:
            List of extracted results.
        """
        url = self.build_url(query, start)
        snapshot = await self.fetcher.fetch(url)
        if snapshot.status_code != 200:
            return []

        if self.client is None:
            return []

        data = await self.client.chat_with_tool(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Extract search results from the following Google result page HTML:\n\n{snapshot.html[:12000]}",
            tool_def=EXTRACT_RESULTS_TOOL,
        )
        items = data.get("results", [])
        results: list[GoogleResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            result_url = item.get("url", "")
            if not isinstance(result_url, str) or not result_url.startswith("http"):
                continue
            results.append(
                GoogleResult(
                    title=str(item.get("title", "")),
                    url=result_url,
                    snippet=str(item.get("snippet", "")),
                )
            )
        return results
