"""
Data Agent - Agent Module

This module provides a LangChain 1.2.0 create_agent based implementation
for Text-to-SQL conversion with middleware support.
"""

from data_agent.agent.agent import create_text2sql_agent
from data_agent.agent.prompts import TEXT2SQL_SYSTEM_PROMPT

__all__ = [
    "create_text2sql_agent",
    "TEXT2SQL_SYSTEM_PROMPT",
]
