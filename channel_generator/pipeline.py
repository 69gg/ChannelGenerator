"""Main discovery pipeline orchestration."""

from datetime import UTC, datetime

from channel_generator.analyzer import ChannelInfo, analyze_page
from channel_generator.config import Settings
from channel_generator.fetcher import Fetcher
from channel_generator.keyword_generator import generate_keywords
from channel_generator.llm_client import LLMClient
from channel_generator.recursive_crawler import RecursiveCrawler
from channel_generator.reporter import generate_report, write_report
from channel_generator.state import State
from channel_generator.summarizer import summarize_channels


class DiscoveryPipeline:
    """End-to-end discovery pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = LLMClient(settings)
        self.fetcher = Fetcher()
        self.state = State(settings.state_path)

    async def run(self) -> list[ChannelInfo]:
        """Run the full discovery pipeline.

        Returns:
            List of newly discovered/verified channels.
        """
        print(f"Generating up to {self.settings.keyword_count} keywords...")
        keywords = await generate_keywords(self.client, self.settings)
        print(f"Generated {len(keywords)} keywords.")

        print("Discovering candidate URLs via recursive crawling...")
        crawler = RecursiveCrawler(self.settings, self.client, self.fetcher)
        candidate_urls = await crawler.discover(keywords)
        print(f"Discovered {len(candidate_urls)} candidate URLs.")

        print("Analyzing candidates...")
        channels: list[ChannelInfo] = []
        for url in candidate_urls:
            snapshot = await self.fetcher.fetch(url)
            if snapshot.status_code != 200:
                continue
            info = await analyze_page(self.client, snapshot)
            if info:
                channels.append(info)

        print(f"Analyzed {len(channels)} free LLM channels.")

        if not self.settings.dry_run:
            self.state.merge(channels)
            self.state.save()

        records = self.state.list_channels(max_results=self.settings.max_results)
        print(f"Generating report with up to {len(records)} channels...")
        summary = await summarize_channels(self.client, self.settings, records)
        report = generate_report(
            keywords=keywords,
            channels=records,
            summary=summary,
            max_results=self.settings.max_results,
        )
        write_report(report, self.settings.report_path)
        print(f"Report written to {self.settings.report_path}")

        return channels

    async def close(self) -> None:
        """Close resources."""
        await self.fetcher.close()


def current_timestamp() -> str:
    """Return current UTC ISO timestamp."""
    return datetime.now(UTC).isoformat()
