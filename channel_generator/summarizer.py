"""LLM-driven summarization and report structure generation."""

from typing import Any

from channel_generator.config import Settings
from channel_generator.llm_client import LLMClient, tool

SUMMARIZE_TOOL = tool(
    name="summarize_channels",
    description="Summarize a list of discovered free LLM/AI chat channels into a report structure.",
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A short executive summary (2-4 sentences) of the findings.",
            },
            "highlights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 notable or unusual channels worth highlighting.",
            },
            "category_counts": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
                "description": "Number of channels per category.",
            },
            "confidence_counts": {
                "type": "object",
                "properties": {
                    "high": {"type": "integer"},
                    "medium": {"type": "integer"},
                    "low": {"type": "integer"},
                },
                "required": ["high", "medium", "low"],
            },
        },
        "required": ["summary", "highlights", "category_counts", "confidence_counts"],
    },
)

SYSTEM_PROMPT = """You are a research analyst summarizing discovered free LLM/AI chat channels.

Produce a concise executive summary, highlight the most notable channels, and return counts by category and confidence level.
"""


def _truncate(text: str, max_len: int = 8000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


async def summarize_channels(
    client: LLMClient,
    settings: Settings,
    channels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize channels using the summary model.

    Args:
        client: LLM client.
        settings: Application settings.
        channels: List of channel records.

    Returns:
        Structured summary data.
    """
    if not channels:
        return {
            "summary": "No channels discovered in this run.",
            "highlights": [],
            "category_counts": {},
            "confidence_counts": {"high": 0, "medium": 0, "low": 0},
        }

    user_prompt = f"Summarize the following discovered channels:\n\n{_truncate(str(channels))}"
    data = await client.chat_with_tool(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_def=SUMMARIZE_TOOL,
        model=settings.effective_summary_model,
        model_role="summary",
    )
    return {
        "summary": str(data.get("summary", "")),
        "highlights": [str(h) for h in data.get("highlights", [])],
        "category_counts": dict(data.get("category_counts", {})),
        "confidence_counts": dict(
            data.get("confidence_counts", {"high": 0, "medium": 0, "low": 0})
        ),
    }
