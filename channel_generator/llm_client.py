"""Async OpenAI-compatible LLM client with tool-call support."""

from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam

from channel_generator.config import Settings


def tool(name: str, description: str, parameters: dict[str, Any]) -> ChatCompletionToolParam:
    """Build an OpenAI tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class LLMClient:
    """Async LLM client wrapper with tool-call support."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "no-key",
        )

    async def chat_with_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_def: ChatCompletionToolParam,
        model: str | None = None,
        temperature: float = 0.5,
    ) -> dict[str, Any]:
        """Call the LLM with a single required tool and return the tool arguments.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.
            tool_def: OpenAI tool definition.
            model: Optional model override; defaults to settings.llm_model.
            temperature: Sampling temperature.

        Returns:
            Parsed tool arguments as a dict.
        """
        response = await self.client.chat.completions.create(
            model=model or self.settings.llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": tool_def["function"]["name"]}},
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("LLM did not return a tool call")
        import json

        return json.loads(message.tool_calls[0].function.arguments)

    async def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.5,
    ) -> str:
        """Call the LLM and return plain text.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.
            model: Optional model override.
            temperature: Sampling temperature.

        Returns:
            Response text.
        """
        response = await self.client.chat.completions.create(
            model=model or self.settings.llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content or ""
