"""LLM-driven keyword generation using tool calls."""

from channel_generator.config import Settings
from channel_generator.llm_client import LLMClient, tool

SYSTEM_PROMPT = """You are a search strategist for discovering websites that offer free LLM chat / AI assistant functionality.

Your task: generate diverse search keywords that will help find such sites across the web.

Guidelines:
- Cover multiple languages (English, Chinese, Japanese, Russian, Spanish, etc.)
- Cover multiple scenarios: pure AI chat, design tools with AI chat, coding assistants, writing/office assistants, roleplay/character chat, AI aggregator platforms, free model trials
- Include expressions like: "free AI chat", "free LLM", "no login AI chat", "free GPT alternative", "免费 AI 对话", "無料 AI チャット", "chatbot gratis", "бесплатный чат GPT", etc.
"""

KEYWORDS_TOOL = tool(
    name="generate_keywords",
    description="Generate a diverse list of search keywords for discovering free LLM/AI chat websites.",
    parameters={
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search keyword strings.",
            },
        },
        "required": ["keywords"],
    },
)


def build_user_prompt(count: int, manual_keywords: list[str]) -> str:
    """Build the user prompt for keyword generation."""
    base = f"Generate exactly {count} diverse search keywords for discovering free LLM/AI chat websites."
    if manual_keywords:
        base += f"\nInclude variations around these user-provided seeds: {', '.join(manual_keywords)}."
    return base


async def generate_keywords(
    client: LLMClient,
    settings: Settings,
) -> list[str]:
    """Generate search keywords via LLM tool call.

    Args:
        client: Async LLM client.
        settings: Application settings.

    Returns:
        List of keyword strings.
    """
    user_prompt = build_user_prompt(settings.keyword_count, settings.manual_keywords)
    data = await client.chat_with_tool(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_def=KEYWORDS_TOOL,
    )
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        raise ValueError(f"LLM returned invalid keywords format: {data}")

    # Deduplicate and merge with manual keywords
    seen: set[str] = set()
    result: list[str] = []
    for kw in list(settings.manual_keywords) + keywords:
        normalized = kw.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(kw.strip())
    return result
