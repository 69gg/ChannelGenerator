"""Command-line interface for ChannelGenerator."""

from pathlib import Path

import typer

from channel_generator.config import load_settings

app = typer.Typer(
    name="channel-generator",
    help="Discover free LLM chat channels across the web.",
    no_args_is_help=True,
)


@app.command()
def run(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a TOML config file.",
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        help="Base URL for the OpenAI-compatible LLM API.",
    ),
    llm_api_key: str | None = typer.Option(
        None,
        "--llm-api-key",
        help="API key for the OpenAI-compatible LLM API.",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Model name for analysis.",
    ),
    llm_summary_model: str | None = typer.Option(
        None,
        "--llm-summary-model",
        help="Model name for final summarization.",
    ),
    keyword_count: int | None = typer.Option(
        None,
        "--keyword-count",
        help="Number of keywords to generate via LLM.",
    ),
    urls_per_search_page: int | None = typer.Option(
        None,
        "--urls-per-search-page",
        help="Number of URLs to pick from each Google search result page.",
    ),
    recursion_depth: int | None = typer.Option(
        None,
        "--recursion-depth",
        help="Maximum recursion depth for URL discovery.",
    ),
    urls_per_recursion: int | None = typer.Option(
        None,
        "--urls-per-recursion",
        help="Number of promising external links to follow per recursion level.",
    ),
    max_search_per_query: int | None = typer.Option(
        None,
        "--max-search-per-query",
        help="Max results per Firecrawl keyless search query.",
    ),
    search_keywords: str | None = typer.Option(
        None,
        "--search-keywords",
        help="Optional comma-separated manual keywords.",
    ),
    max_results: int | None = typer.Option(
        None,
        "--max-results",
        help="Maximum number of channels in the final report.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the generated Markdown report.",
    ),
    state_file: Path | None = typer.Option(
        None,
        "--state-file",
        help="Path to the JSON state file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run discovery without writing report or state.",
    ),
) -> None:
    """Run the full discovery pipeline."""
    overrides = {
        key: value
        for key, value in {
            "llm_base_url": llm_base_url,
            "llm_api_key": llm_api_key,
            "llm_model": llm_model,
            "llm_summary_model": llm_summary_model,
            "keyword_count": keyword_count,
            "urls_per_search_page": urls_per_search_page,
            "recursion_depth": recursion_depth,
            "urls_per_recursion": urls_per_recursion,
            "max_search_per_query": max_search_per_query,
            "search_keywords": search_keywords,
            "max_results": max_results,
            "report_path": output,
            "state_path": state_file,
            "dry_run": dry_run,
        }.items()
        if value is not None
    }
    settings = load_settings(config_path=config, **overrides)
    import asyncio

    from channel_generator.pipeline import DiscoveryPipeline

    async def _run() -> None:
        pipeline = DiscoveryPipeline(settings)
        try:
            await pipeline.run()
        finally:
            await pipeline.close()

    asyncio.run(_run())


@app.command()
def version() -> None:
    """Show the version."""
    from channel_generator import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
