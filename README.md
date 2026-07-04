# ChannelGenerator

AI-driven discovery of free LLM chat channels.

This tool automatically searches the web for websites that offer free LLM/AI chat functionality, analyzes them, and produces a Markdown report. It is designed to run both locally and as a scheduled GitHub Actions workflow.

## What it finds

Any site that offers free LLM-based text chat/assistant functionality, including:

- Pure AI chat sites
- Design tools with embedded AI chat
- Coding assistants with chat
- Writing/office assistants with chat
- AI aggregator platforms
- Roleplay/character chat platforms
- Model vendor free trial pages

## Quick start (local)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run channel-generator run \
  --llm-base-url https://api.openai.com/v1 \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-model gpt-5.5 \
  --llm-summary-model gpt-5.5 \
  --max-results 50
```

Results are written to `report.md` and `state.json`.

You can also use a config file:

```bash
cp config.toml.example config.toml
# edit config.toml
uv run channel-generator run --config config.toml
```

## GitHub Actions

1. Push this repository to GitHub.
2. Go to **Settings > Secrets and variables > Actions** and add:
   - `LLM_API_KEY` (Secret)
3. Add Repository variables (optional):
   - `LLM_BASE_URL` (default: `https://api.openai.com/v1`)
   - `LLM_MODEL` (default: `gpt-5.5`)
   - `LLM_SUMMARY_MODEL` (default: `gpt-5.5`)
   - `KEYWORD_COUNT` (default: `30`)
   - `URLS_PER_SEARCH_PAGE` (default: `5`)
   - `RECURSION_DEPTH` (default: `2`)
   - `URLS_PER_RECURSION` (default: `3`)
   - `MAX_SEARCH_PER_QUERY` (default: `20`)
   - `MAX_RESULTS` (default: `200`)
4. The workflow runs daily at 02:00 UTC and can be triggered manually from the **Actions** tab.
5. Each run creates a GitHub Release with the trigger time as the tag and title, attaching `report.md` and `state.json`.

> Generated artifacts (`report.md`, `state.json`) are ignored by git and only distributed via Releases.

## How it works

1. **Keyword generation**: The LLM generates a diverse set of search keywords in multiple languages and scenarios.
2. **Google search crawling**: Each keyword is submitted to Google and the result page HTML is fetched directly.
3. **LLM URL selection**: The LLM picks the most promising result URLs to investigate.
4. **Recursive crawling**: Each selected URL is fetched. The LLM decides whether it is a free LLM channel and/or which outbound links to follow next, up to the configured recursion depth.
5. **Channel analysis**: Candidate pages are analyzed by the LLM to extract structured metadata (name, models, free tier, category, confidence, etc.).
6. **Summarization & report**: The LLM summarizes the findings and a Markdown report is generated, sorted by recency.

Supplementary sources (Firecrawl keyless search, GitHub, Hacker News, Reddit) are also available in `channel_generator/sources/` for future integration.

## CLI options

```
--config PATH                  Path to a TOML config file
--llm-base-url TEXT            OpenAI-compatible API base URL
--llm-api-key TEXT             OpenAI-compatible API key
--llm-model TEXT               Model for keyword/url/analysis steps
--llm-summary-model TEXT       Model for final summarization
--keyword-count INTEGER        Number of keywords to generate
--urls-per-search-page INT     URLs to pick from each Google result page
--recursion-depth INT          Maximum recursion depth
--urls-per-recursion INT       Promising links to follow per recursion level
--search-keywords TEXT         Optional comma-separated manual keywords
--max-search-per-query INT     Max Firecrawl keyless results per query
--max-results INT              Maximum channels in the report
--output PATH                  Report output path
--state-file PATH              State JSON path
--dry-run                      Run without writing report/state
```

## Development

```bash
uv run ruff check .      # lint
uv run pytest            # tests
```

## License

MIT
