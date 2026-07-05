"""Async OpenAI-compatible LLM client with tool-call support."""

import json
from json import JSONDecodeError
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam

from channel_generator.config import Settings

TOOL_CALL_ATTEMPTS = 2


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
        model_role: Literal["main", "summary"] = "main",
        temperature: float = 0.5,
    ) -> dict[str, Any]:
        """Call the LLM with a single required tool and return the tool arguments.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.
            tool_def: OpenAI tool definition.
            model: Optional model override; defaults to settings.llm_model.
            model_role: Which model option profile to apply.
            temperature: Sampling temperature.

        Returns:
            Parsed tool arguments as a dict.
        """
        last_error = "unknown error"
        for _attempt in range(TOOL_CALL_ATTEMPTS):
            response = await self.client.chat.completions.create(
                model=model or self.settings.llm_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[tool_def],
                tool_choice={"type": "function", "function": {"name": tool_def["function"]["name"]}},
                **self.settings.chat_completion_options(model_role),
            )
            message = response.choices[0].message
            if not message.tool_calls:
                last_error = "LLM did not return a tool call"
                continue

            arguments = message.tool_calls[0].function.arguments
            try:
                data = json.loads(arguments)
            except JSONDecodeError as exc:
                last_error = f"invalid tool-call JSON: {exc.msg}"
                continue

            if not isinstance(data, dict):
                last_error = "tool-call JSON was not an object"
                continue

            return data

        raise ValueError(f"LLM did not return valid tool-call arguments: {last_error}")

    async def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        model_role: Literal["main", "summary"] = "main",
        temperature: float = 0.5,
    ) -> str:
        """Call the LLM and return plain text.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.
            model: Optional model override.
            model_role: Which model option profile to apply.
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
            **self.settings.chat_completion_options(model_role),
        )
        content = response.choices[0].message.content
        return content or ""
