"""LLM-driven channel analysis using tool calls."""

from dataclasses import dataclass
from datetime import UTC, datetime

from channel_generator.fetcher import PageSnapshot
from channel_generator.llm_client import LLMClient, tool


@dataclass
class ChannelInfo:
    """Structured information about a discovered channel."""

    url: str
    name: str
    description: str
    models: list[str]
    free_tier_desc: str
    requires_auth: bool
    category: str
    confidence: str
    notes: str
    first_seen_at: str
    last_verified_at: str


ANALYZE_TOOL = tool(
    name="analyze_channel",
    description="Analyze a web page to determine if it is a free LLM/AI chat channel and extract structured metadata.",
    parameters={
        "type": "object",
        "properties": {
            "is_free_llm_chat": {
                "type": "boolean",
                "description": "Whether the page offers free LLM chat / AI assistant functionality.",
            },
            "name": {
                "type": "string",
                "description": "Product or site name.",
            },
            "description": {
                "type": "string",
                "description": "Short description of what the site offers.",
            },
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Models or providers mentioned (e.g., GPT-4o, Claude, Llama).",
            },
            "free_tier_desc": {
                "type": "string",
                "description": "Description of the free tier or limits.",
            },
            "requires_auth": {
                "type": "boolean",
                "description": "Whether users must sign up or log in to use the free chat.",
            },
            "category": {
                "type": "string",
                "enum": ["pure_chat", "design_tool", "coding_tool", "writing_tool", "aggregator", "roleplay", "other"],
                "description": "Category of the site.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence that this is a free LLM chat channel.",
            },
            "notes": {
                "type": "string",
                "description": "Any risk signals, caveats, or extra notes.",
            },
        },
        "required": [
            "is_free_llm_chat",
            "name",
            "description",
            "models",
            "free_tier_desc",
            "requires_auth",
            "category",
            "confidence",
            "notes",
        ],
    },
)

SYSTEM_PROMPT = """You are evaluating web pages for a research project.

A channel is any website that offers free LLM/AI chat functionality to individual users. This includes:
- Pure AI chat sites
- Design tools with embedded AI chat/assistant
- Coding assistants with chat
- Writing/office assistants with chat
- AI aggregator platforms
- Roleplay/character chat platforms
- Model vendor free trial pages

The site must offer some free tier (no credit card required, daily free quota, free trial, etc.).
Use the analyze_channel tool to report your findings.
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _truncate(text: str, max_len: int = 6000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def analyze_page(
    client: LLMClient,
    snapshot: PageSnapshot,
) -> ChannelInfo | None:
    """Analyze a fetched page and return structured channel info if relevant.

    Args:
        client: LLM client.
        snapshot: Fetched page snapshot.

    Returns:
        ChannelInfo if the page is a free LLM channel, otherwise None.
    """
    user_prompt = (
        f"URL: {snapshot.url}\n"
        f"Title: {snapshot.title}\n"
        f"Meta description: {snapshot.description}\n"
        f"Page text:\n{_truncate(snapshot.text)}\n\n"
        "Analyze this page. If it is not a free LLM/AI chat channel, set is_free_llm_chat=false; "
        "otherwise fill all fields accurately."
    )
    data = await client.chat_with_tool(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_def=ANALYZE_TOOL,
    )
    if not data.get("is_free_llm_chat", False):
        return None

    now = _now_iso()
    return ChannelInfo(
        url=snapshot.url,
        name=str(data.get("name", snapshot.title or "Unknown")),
        description=str(data.get("description", snapshot.description or "")),
        models=[str(m) for m in data.get("models", []) if m],
        free_tier_desc=str(data.get("free_tier_desc", "")),
        requires_auth=bool(data.get("requires_auth", False)),
        category=str(data.get("category", "other")),
        confidence=str(data.get("confidence", "low")),
        notes=str(data.get("notes", "")),
        first_seen_at=now,
        last_verified_at=now,
    )
