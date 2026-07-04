"""Hacker News source via Algolia search API."""


import httpx

from channel_generator.fetcher import default_headers


class HackerNewsSource:
    """Search Hacker News and extract URLs from stories/comments."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.client = httpx.AsyncClient(
            headers=default_headers(),
            timeout=timeout,
        )

    async def search(self, query: str, hits_per_page: int = 20) -> list[str]:
        """Search Hacker News via Algolia and return URLs.

        Args:
            query: Search query.
            hits_per_page: Number of hits to fetch.

        Returns:
            List of URLs.
        """
        try:
            response = await self.client.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={"query": query, "hitsPerPage": hits_per_page},
            )
            response.raise_for_status()
        except Exception:
            return []

        payload = response.json()
        hits = payload.get("hits", [])
        urls: list[str] = []
        for hit in hits:
            url = hit.get("url")
            if url and isinstance(url, str) and url.startswith("http"):
                urls.append(url)
            story_url = hit.get("story_url")
            if story_url and isinstance(story_url, str) and story_url.startswith("http"):
                urls.append(story_url)
        return urls

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
