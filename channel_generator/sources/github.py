"""GitHub source for discovering channels from awesome lists and search."""

import httpx

from channel_generator.fetcher import default_headers


class GitHubSource:
    """Search GitHub and extract URLs from README contents."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.client = httpx.AsyncClient(
            headers=default_headers(),
            timeout=timeout,
        )

    async def search(self, query: str, per_page: int = 10) -> list[str]:
        """Search GitHub repositories and return URLs found in READMEs.

        Args:
            query: Search query.
            per_page: Max repositories to inspect.

        Returns:
            List of extracted URLs.
        """
        try:
            response = await self.client.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": per_page, "sort": "updated"},
            )
            response.raise_for_status()
        except Exception:
            return []

        payload = response.json()
        items = payload.get("items", [])
        urls: list[str] = []
        for item in items:
            raw_url = item.get("html_url", "").replace(
                "https://github.com", "https://raw.githubusercontent.com"
            )
            raw_url += "/main/README.md"
            readme_text = await self._fetch_raw(raw_url)
            if not readme_text:
                # try master branch fallback
                raw_url = raw_url.replace("/main/README.md", "/master/README.md")
                readme_text = await self._fetch_raw(raw_url)
            urls.extend(self._extract_urls(readme_text))
        return urls

    async def _fetch_raw(self, raw_url: str) -> str:
        """Fetch raw README content."""
        try:
            response = await self.client.get(raw_url)
            response.raise_for_status()
            return response.text
        except Exception:
            return ""

    def _extract_urls(self, text: str) -> list[str]:
        """Extract markdown links from README text."""
        import re

        pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
        found: list[str] = []
        for match in pattern.findall(text):
            url = match[1]
            if "github.com" in url or url.endswith("README.md"):
                continue
            found.append(url)
        return found

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
