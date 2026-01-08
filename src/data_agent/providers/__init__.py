"""
LLM Providers Module

This module provides provider-agnostic LLM integration using LangChain's
chat model abstractions. Defaults to OpenAI but supports easy switching.
"""

from data_agent.providers.base import get_llm, LLMConfig

__all__ = [
    "get_llm",
    "LLMConfig",
]
