"""HTTP fetching and HTML parsing utilities."""

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 20.0
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class PageSnapshot:
    """Snapshot of a fetched page."""

    url: str
    status_code: int
    title: str
    description: str
    text: str
    links: list[str]
    html: str = ""


class Fetcher:
    """Async HTTP fetcher with polite defaults."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    async def fetch(self, url: str) -> PageSnapshot:
        """Fetch and parse a URL.

        Args:
            url: Target URL.

        Returns:
            Parsed page snapshot.
        """
        try:
            response = await self.client.get(url)
        except Exception as exc:
            return PageSnapshot(
                url=url,
                status_code=0,
                title="",
                description=str(exc),
                text="",
                links=[],
            )

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and response.text:
            # Some search pages may omit proper content-type
            pass

        text = response.text
        soup = BeautifulSoup(text, "lxml")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc.get("content", "")
        else:
            meta_og = soup.find("meta", attrs={"property": "og:description"})
            if meta_og and meta_og.get("content"):
                description = meta_og.get("content", "")

        # Visible body text (truncated later by caller if needed)
        body = soup.find("body")
        body_text = body.get_text(separator="\n", strip=True) if body else ""

        links: list[str] = []
        base_url = str(response.url)
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in ("http", "https"):
                links.append(absolute)

        return PageSnapshot(
            url=base_url,
            status_code=response.status_code,
            title=title,
            description=description,
            text=body_text,
            links=links,
            html=text,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
