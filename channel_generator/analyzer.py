"""LLM-driven channel analysis using tool calls with chunked parallel processing."""

import asyncio
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

KNOWN_MODELS = [
    "gpt", "gpt-5", "gpt-5.5", "gpt-5.5-mini",
    "claude", "claude 4.8", "claude 5", "claude-4.8", "claude-5",
    "gemini", "gemini 2.5", "gemini-2.5",
    "glm", "glm-5", "glm-5.2", "glm5.2",
    "kimi", "kimi k2.6", "kimi k2.7", "k2.6", "k2.7",
    "deepseek", "deepseek v4", "deepseek-v4",
    "llama", "qwen", "yi", "baichuan", "mixtral",
]

SYSTEM_PROMPT = f"""You are evaluating web pages for a research project.

A channel is any website that offers free LLM/AI chat functionality to individual users. This includes:
- Pure AI chat sites
- Design tools with embedded AI chat/assistant
- Coding assistants with chat
- Writing/office assistants with chat
- AI aggregator platforms
- Roleplay/character chat platforms
- Model vendor free trial pages
- AI-driven apps, smart design tools, and any tool with a free AI assistant/chat feature

The site must offer some free tier (no credit card required, daily free quota, free trial, etc.).

Well-known models/providers to look for: {', '.join(KNOWN_MODELS)}.
Use the analyze_channel tool to report your findings.
"""

CHUNK_SIZE = 20000
MAX_CHUNKS = 5


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _chunks(text: str, size: int) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    step = size // 2
    start = 0
    while start < len(text) and len(pieces) < MAX_CHUNKS:
        pieces.append(text[start : start + size])
        start += step
    return pieces


async def _analyze_chunk(
    client: LLMClient,
    url: str,
    title: str,
    description: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    """Analyze a single text chunk."""
    user_prompt = (
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Meta description: {description}\n"
        f"This is chunk {chunk_index + 1} of {total_chunks} of the page text.\n\n"
        f"Page text chunk:\n{chunk_text}\n\n"
        "Analyze this chunk. If the page is not a free LLM/AI chat channel, set is_free_llm_chat=false; "
        "otherwise fill all fields accurately. Use only evidence visible in this chunk."
    )
    try:
        return await client.chat_with_tool(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tool_def=ANALYZE_TOOL,
        )
    except Exception as exc:
        return {"is_free_llm_chat": False, "error": str(exc)}


def _merge_analyses(analyses: list[dict], url: str, title: str, description: str) -> ChannelInfo | None:
    """Merge chunk analyses into a single ChannelInfo."""
    positive = [a for a in analyses if a.get("is_free_llm_chat", False)]
    if not positive:
        return None

    # Use the highest-confidence or first positive result as base.
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    base = max(
        positive,
        key=lambda a: confidence_order.get(a.get("confidence", "low"), 0),
    )

    # Collect all models mentioned across chunks.
    all_models: set[str] = set()
    for a in positive:
        for m in a.get("models", []):
            if m:
                all_models.add(str(m).strip())

    # Collect notes and concatenate distinct ones.
    notes_parts: list[str] = []
    for a in positive:
        note = a.get("notes", "")
        if note and note not in notes_parts:
            notes_parts.append(str(note))

    now = _now_iso()
    return ChannelInfo(
        url=url,
        name=str(base.get("name", title or "Unknown")),
        description=str(base.get("description", description or "")),
        models=sorted(all_models),
        free_tier_desc=str(base.get("free_tier_desc", "")),
        requires_auth=bool(base.get("requires_auth", False)),
        category=str(base.get("category", "other")),
        confidence=str(base.get("confidence", "low")),
        notes="; ".join(notes_parts),
        first_seen_at=now,
        last_verified_at=now,
    )


async def analyze_page(
    client: LLMClient,
    snapshot: PageSnapshot,
) -> ChannelInfo | None:
    """Analyze a fetched page and return structured channel info if relevant.

    Long pages are split into chunks and analyzed in parallel.

    Args:
        client: LLM client.
        snapshot: Fetched page snapshot.

    Returns:
        ChannelInfo if the page is a free LLM channel, otherwise None.
    """
    text_chunks = _chunks(snapshot.text, CHUNK_SIZE)
    tasks = [
        _analyze_chunk(
            client,
            snapshot.url,
            snapshot.title,
            snapshot.description,
            chunk,
            idx,
            len(text_chunks),
        )
        for idx, chunk in enumerate(text_chunks)
    ]
    analyses = await asyncio.gather(*tasks, return_exceptions=True)
    analyses = [a for a in analyses if not isinstance(a, Exception)]

    if not analyses:
        return None

    return _merge_analyses(analyses, snapshot.url, snapshot.title, snapshot.description)
