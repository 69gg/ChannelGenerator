"""Multi-agent LLM-driven keyword generation."""

import asyncio

from channel_generator.config import Settings
from channel_generator.llm_client import LLMClient, tool

KEYWORDS_TOOL = tool(
    name="generate_keywords",
    description="Generate a list of search keywords from a specific perspective.",
    parameters={
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search keyword strings from this perspective.",
            },
        },
        "required": ["keywords"],
    },
)

PERSPECTIVES = [
    {
        "name": "general_free_llm",
        "prompt": (
            "Generate keywords for finding general free LLM chat websites and GPT alternatives. "
            "Cover multiple languages. Examples: free AI chat, free LLM, free GPT alternative, "
            "no login AI chat, 免费 AI 对话, 無料 AI チャット, chatbot gratis, бесплатный чат GPT."
        ),
    },
    {
        "name": "ai_driven_apps",
        "prompt": (
            "Generate keywords for finding AI-driven applications and productivity tools that "
            "include free AI chat/assistant features. Examples: AI-powered design tool free, "
            "AI driven app free, 智能设计 免费, AI assistant free, free AI copilot."
        ),
    },
    {
        "name": "coding_assistants",
        "prompt": (
            "Generate keywords for finding free coding assistants and AI programming tools with chat. "
            "Examples: free AI coding assistant, free code copilot, AI programmer free, "
            "免费 AI 编程助手, free coding AI chat."
        ),
    },
    {
        "name": "writing_office",
        "prompt": (
            "Generate keywords for finding free AI writing assistants, office tools, and content generators. "
            "Examples: free AI writing assistant, free AI content generator, AI writer free, "
            "免费 AI 写作助手, free AI document assistant."
        ),
    },
    {
        "name": "roleplay_character",
        "prompt": (
            "Generate keywords for finding free AI roleplay and character chat platforms. "
            "Examples: free AI character chat, free roleplay AI, AI companion free, "
            "免费 AI 角色扮演, free AI chatbot character."
        ),
    },
    {
        "name": "aggregators_and_trials",
        "prompt": (
            "Generate keywords for finding AI aggregator platforms and model vendor free trials. "
            "Examples: free AI model playground, LLM aggregator free, free Claude trial, "
            "free GPT-4o access, AI model comparison free."
        ),
    },
    {
        "name": "image_multimodal",
        "prompt": (
            "Generate keywords for finding free AI image generation or multimodal tools that also "
            "offer chat/assistant features. Examples: free AI image generator chat, free AI art bot, "
            "multimodal AI chat free, 免费 AI 绘画 聊天."
        ),
    },
    {
        "name": "regional_niche",
        "prompt": (
            "Generate keywords for finding regional, niche, or newly launched free AI chat services. "
            "Cover non-English markets and long-tail phrases. Examples: 免费聊天机器人, "
            "chat IA gratuit, chat de IA gratis, yapay zeka sohbet ücretsiz, free AI chat no sign up."
        ),
    },
]


def build_user_prompt(count_per_agent: int, manual_keywords: list[str]) -> str:
    """Build the user prompt for a single keyword agent."""
    base = f"Generate exactly {count_per_agent} search keywords from your perspective."
    if manual_keywords:
        base += f"\nConsider these user-provided seeds: {', '.join(manual_keywords)}."
    return base


async def _generate_from_perspective(
    client: LLMClient,
    perspective: dict[str, str],
    count_per_agent: int,
    manual_keywords: list[str],
) -> list[str]:
    """Run one keyword-generation agent."""
    data = await client.chat_with_tool(
        system_prompt=perspective["prompt"],
        user_prompt=build_user_prompt(count_per_agent, manual_keywords),
        tool_def=KEYWORDS_TOOL,
    )
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        return []
    return [str(k).strip() for k in keywords if str(k).strip()]


async def generate_keywords(
    client: LLMClient,
    settings: Settings,
) -> list[str]:
    """Generate search keywords via multiple parallel LLM agents.

    Args:
        client: Async LLM client.
        settings: Application settings.

    Returns:
        List of deduplicated keyword strings.
    """
    num_agents = max(1, min(settings.keyword_agents, len(PERSPECTIVES)))
    perspectives = PERSPECTIVES[:num_agents]
    # Reserve slots for manual keywords and distribute the rest among agents.
    manual = settings.manual_keywords
    remaining = max(settings.keyword_count - len(manual), 0)
    count_per_agent = max(1, remaining // num_agents)

    tasks = [_generate_from_perspective(client, p, count_per_agent, manual) for p in perspectives]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen: set[str] = set()
    output: list[str] = []
    for kw in manual:
        normalized = kw.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(kw.strip())
    for batch in results:
        if isinstance(batch, Exception):
            continue
        for kw in batch:
            normalized = kw.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(kw.strip())
    return output
