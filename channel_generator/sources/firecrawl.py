"""Firecrawl keyless search API as a supplementary source."""

from dataclasses import dataclass

import httpx

from channel_generator.fetcher import DEFAULT_HEADERS


@dataclass
class FirecrawlResult:
    """A single Firecrawl search result."""

    title: str
    url: str
    description: str
    category: str


class FirecrawlSource:
    """Keyless Firecrawl /v2/search source."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
        )

    async def search(self, query: str, limit: int = 10) -> list[FirecrawlResult]:
        """Search using Firecrawl keyless endpoint.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of results.
        """
        try:
            response = await self.client.post(
                "https://api.firecrawl.dev/v2/search",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
        except Exception:
            return []

        payload = response.json()
        data = payload.get("data", {})
        web = data.get("web", [])
        results: list[FirecrawlResult] = []
        for item in web:
            if not isinstance(item, dict):
                continue
            url = item.get("url", "")
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            results.append(
                FirecrawlResult(
                    title=str(item.get("title", "")),
                    url=url,
                    description=str(item.get("description", "")),
                    category=str(item.get("category", "")),
                )
            )
        return results

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
