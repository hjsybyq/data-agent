"""
Vanna LangGraph Builder

This module constructs the core reasoning graph for text-to-SQL conversion.
"""

from typing import Optional, Any, Sequence, Callable
from langgraph.graph import StateGraph, START, END

from data_agent.graph.state import AgentState
from data_agent.graph.nodes import (
    input_normalization,
    schema_acquisition,
    sql_generation,
    sql_validation,
    sql_execution,
    result_evaluation,
    # Question decomposition nodes
    question_analysis,
    sub_query_executor,
    result_synthesizer,
    # RAG nodes
    example_retrieval,
)
from data_agent.graph.edges import (
    should_retry_sql,
    is_sql_valid,
    execution_succeeded,
    route_by_complexity,
    has_more_sub_queries,
)


def create_agent_graph(
    checkpointer: Optional[Any] = None,
    enable_decomposition: bool = True,
    enable_rag: bool = True,
) -> StateGraph:
    """
    Create the Vanna text-to-SQL reasoning graph.
    
    The graph implements the following flow:
    
    1. Input Normalization - Clean and standardize user question
    2. Question Analysis - Determine if question is simple or complex
       - Simple questions -> Schema Acquisition -> SQL Generation flow
       - Complex questions -> Decomposition -> Sub-query loop -> Synthesis
    3. Schema Acquisition - Get relevant database schema
    4. SQL Generation - Generate SQL from question using LLM
    5. SQL Validation - Validate SQL syntax and security
       - If invalid and retries remaining -> back to SQL Generation
       - If invalid and no retries -> to Result Evaluation (error)
    6. SQL Execution - Execute the validated SQL
       - If error and retries remaining -> back to SQL Generation
       - If success -> to Result Evaluation
    7. Result Evaluation - Generate natural language response
    
    For complex questions (with decomposition enabled):
    - Sub-query Executor loops through each sub-question
    - Result Synthesizer combines all sub-results
    
    Args:
        checkpointer: Optional LangGraph checkpointer for persistence
        enable_decomposition: Whether to enable complex question decomposition
        enable_rag: Whether to enable RAG-based example retrieval
        
    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the graph with AgentState
    graph = StateGraph(AgentState)
    
    # Add core nodes
    graph.add_node("input_normalization", input_normalization)
    graph.add_node("schema_acquisition", schema_acquisition)
    graph.add_node("sql_generation", sql_generation)
    graph.add_node("sql_validation", sql_validation)
    graph.add_node("sql_execution", sql_execution)
    graph.add_node("result_evaluation", result_evaluation)
    
    # Add decomposition nodes
    if enable_decomposition:
        graph.add_node("question_analysis", question_analysis)
        graph.add_node("sub_query_executor", sub_query_executor)
        graph.add_node("result_synthesizer", result_synthesizer)
    
    # Start with input normalization
    graph.add_edge(START, "input_normalization")
    
    if enable_decomposition:
        # After normalization, analyze question complexity
        graph.add_edge("input_normalization", "question_analysis")
        
        # Route based on complexity
        graph.add_conditional_edges(
            "question_analysis",
            route_by_complexity,
            {
                "simple": "schema_acquisition",
                "complex": "sub_query_executor",
            }
        )
        
        # Sub-query loop: continue executing or synthesize
        graph.add_conditional_edges(
            "sub_query_executor",
            has_more_sub_queries,
            {
                "continue": "sub_query_executor",
                "synthesize": "result_synthesizer",
            }
        )
        
        # Result synthesizer ends the graph for complex questions
        graph.add_edge("result_synthesizer", END)
    else:
        # Without decomposition, go directly to schema acquisition
        graph.add_edge("input_normalization", "schema_acquisition")
    
    # Add RAG node if enabled
    if enable_rag:
        graph.add_node("example_retrieval", example_retrieval)
        graph.add_edge("schema_acquisition", "example_retrieval")
        graph.add_edge("example_retrieval", "sql_generation")
    else:
        # Without RAG, go directly from schema to sql_generation
        graph.add_edge("schema_acquisition", "sql_generation")
    
    # After SQL generation, always validate
    graph.add_edge("sql_generation", "sql_validation")
    
    # Conditional: after validation
    graph.add_conditional_edges(
        "sql_validation",
        lambda state: (
            "execute" if state.get("is_sql_valid") 
            else ("retry" if should_retry_sql(state) == "retry" else "evaluate")
        ),
        {
            "execute": "sql_execution",
            "retry": "sql_generation",
            "evaluate": "result_evaluation",
        }
    )
    
    # Conditional: after execution
    graph.add_conditional_edges(
        "sql_execution",
        lambda state: (
            "evaluate" if execution_succeeded(state) == "success"
            else ("retry" if should_retry_sql(state) == "retry" else "evaluate")
        ),
        {
            "evaluate": "result_evaluation",
            "retry": "sql_generation",
        }
    )
    
    # Result evaluation ends the graph for simple questions
    graph.add_edge("result_evaluation", END)
    
    # Compile the graph
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    
    return graph.compile()


def create_simple_graph() -> StateGraph:
    """
    Create a simplified graph for basic text-to-SQL without retry loops.
    
    This is useful for quick testing or simpler use cases where
    error correction is handled externally.
    
    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("input_normalization", input_normalization)
    graph.add_node("schema_acquisition", schema_acquisition)
    graph.add_node("sql_generation", sql_generation)
    graph.add_node("result_evaluation", result_evaluation)
    
    # Linear flow without validation/retry
    graph.add_edge(START, "input_normalization")
    graph.add_edge("input_normalization", "schema_acquisition")
    graph.add_edge("schema_acquisition", "sql_generation")
    graph.add_edge("sql_generation", "result_evaluation")
    graph.add_edge("result_evaluation", END)
    
    return graph.compile()
