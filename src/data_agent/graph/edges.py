"""
Conditional Edge Functions for the Vanna LangGraph

These functions are used as routing conditions in the graph to determine
which node to execute next based on the current state.
"""

from typing import Literal
from data_agent.graph.state import AgentState


def should_retry_sql(state: AgentState) -> Literal["retry", "give_up"]:
    """
    Determine whether to retry SQL generation after validation failure.
    
    Args:
        state: Current graph state
        
    Returns:
        "retry" if under max_retries, "give_up" otherwise
    """
    if state["retry_count"] < state["max_retries"]:
        return "retry"
    return "give_up"


def is_sql_valid(state: AgentState) -> Literal["valid", "invalid"]:
    """
    Check if the generated SQL passed validation.
    
    Args:
        state: Current graph state
        
    Returns:
        "valid" if SQL is valid, "invalid" otherwise
    """
    if state.get("is_sql_valid", False):
        return "valid"
    return "invalid"


def execution_succeeded(state: AgentState) -> Literal["success", "error"]:
    """
    Check if SQL execution was successful.
    
    Args:
        state: Current graph state
        
    Returns:
        "success" if no execution error, "error" otherwise
    """
    if state.get("execution_error") is None and state.get("sql_result") is not None:
        sql_result = state["sql_result"]
        if isinstance(sql_result, dict) and sql_result.get("success", False):
            return "success"
    return "error"


def should_terminate(state: AgentState) -> Literal["end", "continue"]:
    """
    Check if the graph should terminate.
    
    Args:
        state: Current graph state
        
    Returns:
        "end" if should_terminate flag is set, "continue" otherwise
    """
    if state.get("should_terminate", False):
        return "end"
    return "continue"


def can_execute_sql(state: AgentState) -> Literal["execute", "skip"]:
    """
    Determine if we have valid SQL to execute.
    
    Args:
        state: Current graph state
        
    Returns:
        "execute" if SQL is valid and present, "skip" otherwise
    """
    if state.get("generated_sql") and state.get("is_sql_valid", False):
        return "execute"
    return "skip"


# ============================================================================
# Question Decomposition Edge Functions
# ============================================================================

def route_by_complexity(state: AgentState) -> Literal["simple", "complex"]:
    """
    Route based on question complexity analysis result.
    
    Args:
        state: Current graph state with question_analysis
        
    Returns:
        "simple" for single-query flow, "complex" for decomposition flow
    """
    analysis = state.get("question_analysis", {})
    if analysis and analysis.get("requires_decomposition", False):
        return "complex"
    return "simple"


def has_more_sub_queries(state: AgentState) -> Literal["continue", "synthesize"]:
    """
    Check if there are more sub-queries to process.
    
    Args:
        state: Current graph state with sub_questions and current_sub_index
        
    Returns:
        "continue" if more sub-queries remain, "synthesize" if all done
    """
    sub_questions = state.get("sub_questions", [])
    current_index = state.get("current_sub_index", 0)
    
    if sub_questions and current_index < len(sub_questions):
        return "continue"
    return "synthesize"

