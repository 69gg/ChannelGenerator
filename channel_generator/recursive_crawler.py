"""Recursive crawler for discovering candidate channels."""

import asyncio

from channel_generator.config import Settings
from channel_generator.fetcher import Fetcher
from channel_generator.llm_client import LLMClient
from channel_generator.sources.bing_search import BingSearchSource
from channel_generator.sources.firecrawl import FirecrawlSource
from channel_generator.sources.google_search import GoogleResult, GoogleSearchSource
from channel_generator.url_selector import (
    CandidateUrl,
    evaluate_page_and_select_links,
    select_from_search_results,
)


class RecursiveCrawler:
    """Crawl starting from search results, recursing through promising links."""

    def __init__(
        self,
        settings: Settings,
        client: LLMClient,
        fetcher: Fetcher,
        firecrawl: FirecrawlSource | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.fetcher = fetcher
        self.google = GoogleSearchSource(fetcher, client)
        self.bing = BingSearchSource(fetcher, client)
        self.firecrawl = firecrawl

    async def _search_all(self, keyword: str) -> tuple[list[GoogleResult], list[GoogleResult]]:
        """Query Google and Bing in parallel for a keyword.

        Args:
            keyword: Search keyword.

        Returns:
            Tuple of (google_results, bing_results_as_google_result).
        """
        google_result, bing_result = await asyncio.gather(
            self.google.search(keyword),
            self.bing.search(keyword),
            return_exceptions=True,
        )
        google_results = [] if isinstance(google_result, Exception) else google_result
        bing_results = [] if isinstance(bing_result, Exception) else bing_result
        if isinstance(google_result, Exception) and isinstance(bing_result, Exception):
            raise RuntimeError(f"Google and Bing search failed: {google_result}; {bing_result}")
        unified_bing = [
            GoogleResult(title=r.title, url=r.url, snippet=r.snippet) for r in bing_results
        ]
        return google_results, unified_bing

    async def _discover_keyword(
        self,
        keyword: str,
        visited: set[str],
        channel_urls: set[str],
        search_sem: asyncio.Semaphore,
        crawl_sem: asyncio.Semaphore,
    ) -> None:
        """Process one keyword and crawl selected URLs."""
        async with search_sem:
            try:
                google_results, bing_results = await self._search_all(keyword)
            except Exception as exc:
                print(f"Skipping keyword {keyword!r}: search failed: {exc}")
                return
            results = google_results + bing_results
            if not results and self.firecrawl is not None:
                try:
                    fc_results = await self.firecrawl.search(
                        keyword, limit=self.settings.max_search_per_query
                    )
                except Exception as exc:
                    print(f"Skipping keyword {keyword!r}: fallback search failed: {exc}")
                    return
                results = [
                    GoogleResult(title=r.title, url=r.url, snippet=r.description)
                    for r in fc_results
                ]
            candidates = [
                CandidateUrl(title=r.title, url=r.url, context=r.snippet) for r in results
            ]
            try:
                selected = await select_from_search_results(
                    self.client,
                    candidates,
                    self.settings.urls_per_search_page,
                )
            except Exception as exc:
                print(f"Skipping keyword {keyword!r}: URL selection failed: {exc}")
                return

        await asyncio.gather(
            *(
                self._crawl(
                    url,
                    depth=0,
                    visited=visited,
                    channel_urls=channel_urls,
                    crawl_sem=crawl_sem,
                )
                for url in selected
            ),
            return_exceptions=True,
        )

    async def discover(self, keywords: list[str]) -> list[str]:
        """Discover candidate channel URLs from a list of keywords.

        Args:
            keywords: Search keywords.

        Returns:
            List of discovered channel URLs.
        """
        channel_urls: set[str] = set()
        visited: set[str] = set()
        search_sem = asyncio.Semaphore(self.settings.effective_concurrency)
        crawl_sem = asyncio.Semaphore(self.settings.effective_concurrency)

        await asyncio.gather(
            *(
                self._discover_keyword(
                    keyword,
                    visited=visited,
                    channel_urls=channel_urls,
                    search_sem=search_sem,
                    crawl_sem=crawl_sem,
                )
                for keyword in keywords
            ),
            return_exceptions=True,
        )

        return list(channel_urls)

    async def _crawl(
        self,
        url: str,
        depth: int,
        visited: set[str],
        channel_urls: set[str],
        crawl_sem: asyncio.Semaphore,
    ) -> None:
        """Recursively crawl a URL.

        Args:
            url: URL to crawl.
            depth: Current recursion depth.
            visited: Set of already visited URLs.
            channel_urls: Accumulator for discovered channel URLs.
        """
        if url in visited:
            return
        visited.add(url)

        async with crawl_sem:
            snapshot = await self.fetcher.fetch(url)
            if snapshot.status_code != 200:
                return

            try:
                is_channel, follow_urls = await evaluate_page_and_select_links(
                    self.client,
                    url=snapshot.url,
                    title=snapshot.title,
                    description=snapshot.description,
                    page_text=snapshot.text,
                    links=snapshot.links,
                    max_follow=self.settings.urls_per_recursion,
                )
            except Exception as exc:
                print(f"Skipping URL {snapshot.url}: page evaluation failed: {exc}")
                return

        if is_channel:
            channel_urls.add(snapshot.url)

        if depth >= self.settings.recursion_depth:
            return

        await asyncio.gather(
            *(
                self._crawl(next_url, depth + 1, visited, channel_urls, crawl_sem)
                for next_url in follow_urls
            ),
            return_exceptions=True,
        )
