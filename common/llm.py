"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("9ROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("9ROUTER_API_KEY"),
        openai_api_base="http://localhost:20128/v1",
        temperature=temperature,
    )
    # return ChatOpenAI(
    #     model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
    #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     temperature=temperature,
    # )