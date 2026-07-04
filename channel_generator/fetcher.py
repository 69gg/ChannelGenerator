"""HTTP fetching and HTML parsing utilities."""

import random
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 20.0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]


def default_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
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
    """Async HTTP fetcher with polite defaults and retries."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, retries: int = 2) -> None:
        self.timeout = timeout
        self.retries = retries
        self.client = httpx.AsyncClient(
            headers=default_headers(),
            timeout=timeout,
            follow_redirects=True,
        )

    async def fetch(self, url: str) -> PageSnapshot:
        """Fetch and parse a URL with retries.

        Args:
            url: Target URL.

        Returns:
            Parsed page snapshot.
        """
        last_error = ""
        for _attempt in range(self.retries + 1):
            try:
                response = await self.client.get(url, headers=default_headers())
                if response.status_code < 500:
                    return self._parse(response)
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = str(exc)
        return PageSnapshot(
            url=url,
            status_code=0,
            title="",
            description=last_error,
            text="",
            links=[],
        )

    def _parse(self, response: httpx.Response) -> PageSnapshot:
        """Parse a successful response into a PageSnapshot."""
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
