"""Recursive crawler for discovering candidate channels."""

from channel_generator.config import Settings
from channel_generator.fetcher import Fetcher
from channel_generator.llm_client import LLMClient
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
        self.firecrawl = firecrawl

    async def discover(self, keywords: list[str]) -> list[str]:
        """Discover candidate channel URLs from a list of keywords.

        Args:
            keywords: Search keywords.

        Returns:
            List of discovered channel URLs.
        """
        channel_urls: set[str] = set()
        visited: set[str] = set()

        for keyword in keywords:
            results = await self.google.search(keyword)
            if not results and self.firecrawl is not None:
                fc_results = await self.firecrawl.search(
                    keyword, limit=self.settings.max_search_per_query
                )
                results = [GoogleResult(title=r.title, url=r.url, snippet=r.description) for r in fc_results]
            candidates = [
                CandidateUrl(title=r.title, url=r.url, context=r.snippet)
                for r in results
            ]
            selected = await select_from_search_results(
                self.client,
                candidates,
                self.settings.urls_per_search_page,
            )

            for url in selected:
                await self._crawl(url, depth=0, visited=visited, channel_urls=channel_urls)

        return list(channel_urls)

    async def _crawl(
        self,
        url: str,
        depth: int,
        visited: set[str],
        channel_urls: set[str],
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

        snapshot = await self.fetcher.fetch(url)
        if snapshot.status_code != 200:
            return

        is_channel, follow_urls = await evaluate_page_and_select_links(
            self.client,
            url=snapshot.url,
            title=snapshot.title,
            description=snapshot.description,
            page_text=snapshot.text,
            links=snapshot.links,
            max_follow=self.settings.urls_per_recursion,
        )

        if is_channel:
            channel_urls.add(snapshot.url)

        if depth >= self.settings.recursion_depth:
            return

        for next_url in follow_urls:
            await self._crawl(next_url, depth + 1, visited, channel_urls)
