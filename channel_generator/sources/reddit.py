"""Reddit source for discovering URLs from public search/posts."""

from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from channel_generator.fetcher import DEFAULT_HEADERS


class RedditSource:
    """Crawl Reddit search results and extract post URLs."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
        )

    async def search(self, query: str, limit: int = 20) -> list[str]:
        """Search Reddit and extract URLs from post listings.

        Args:
            query: Search query.
            limit: Number of posts to inspect.

        Returns:
            List of URLs.
        """
        encoded = quote_plus(query)
        url = f"https://www.reddit.com/search/?q={encoded}"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(response.text, "lxml")
        urls: list[str] = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith("/r/"):
                post_url = f"https://www.reddit.com{href.split('?')[0]}"
                if post_url not in urls:
                    urls.append(post_url)
            elif href.startswith("http") and "reddit.com" not in href:
                urls.append(href)
        return urls[:limit]

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
