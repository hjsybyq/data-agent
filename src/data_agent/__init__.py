"""
Data Agent - A LangGraph-based Text-to-SQL System

This package provides a graph-driven approach to natural language to SQL conversion,
built on LangChain and LangGraph for explicit state management and flow control.
"""

__version__ = "0.1.0"

from data_agent.graph.builder import create_agent_graph
from data_agent.graph.state import AgentState
from data_agent.adapters.data_adapter import DataAgent

__all__ = [
    "create_agent_graph",
    "AgentState",
    "DataAgent",
]
