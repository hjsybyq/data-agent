"""
AgentState - Core State Schema for LangGraph

This module defines the strongly-typed state schema that flows through
the Vanna reasoning graph. All state transitions are explicit and traceable.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class SQLResult(BaseModel):
    """Structured result from SQL execution."""
    
    success: bool = Field(description="Whether the SQL execution succeeded")
    data: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Result data as list of row dictionaries"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if execution failed"
    )
    row_count: int = Field(default=0, description="Number of rows returned")
    columns: List[str] = Field(
        default_factory=list, 
        description="Column names in result"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [{"id": 1, "name": "Alice"}],
                "row_count": 1,
                "columns": ["id", "name"],
            }
        }


class CorrectionEntry(BaseModel):
    """Record of a SQL correction attempt."""
    
    original_sql: str = Field(description="The SQL that failed")
    error: str = Field(description="The error that occurred")
    corrected_sql: Optional[str] = Field(
        default=None, 
        description="The corrected SQL if available"
    )
    attempt_number: int = Field(description="Which retry attempt this was")


class SubQuestion(BaseModel):
    """A sub-question decomposed from a complex question."""
    
    id: str = Field(description="Unique identifier for this sub-question")
    question: str = Field(description="The sub-question text")
    purpose: str = Field(description="What this sub-question aims to find out")
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of sub-questions this one depends on"
    )
    sql: Optional[str] = Field(default=None, description="Generated SQL for this sub-question")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    completed: bool = Field(default=False, description="Whether this sub-question is done")


class QuestionAnalysis(BaseModel):
    """Analysis result for question complexity."""
    
    complexity: str = Field(
        description="Question complexity: 'simple' or 'complex'"
    )
    reasoning: str = Field(description="Why this complexity was determined")
    requires_decomposition: bool = Field(
        default=False,
        description="Whether question needs to be split into sub-questions"
    )
    sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="List of sub-questions if decomposition is needed"
    )


class AgentState(TypedDict):
    """
    Core state schema for the Vanna LangGraph reasoning flow.
    
    This state flows through all nodes in the graph, with each node
    reading relevant fields and updating specific fields as output.
    
    The state is designed to support:
    - Multi-turn conversation via messages
    - Schema-aware SQL generation
    - Iterative error correction with retry loops
    - Traceability of the reasoning process
    """
    
    # Messages for LangGraph conversation tracking
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Input processing
    user_question: str
    normalized_question: Optional[str]
    
    # Conversation context (for multi-turn support)
    conversation_context: Optional[str]
    
    # Schema context
    database_schema: Optional[str]
    table_metadata: Optional[Dict[str, Any]]
    relevant_tables: Optional[List[str]]
    
    # SQL generation
    generated_sql: Optional[str]
    sql_validation_result: Optional[Dict[str, Any]]
    is_sql_valid: bool
    
    # Execution
    sql_result: Optional[Dict[str, Any]]
    execution_error: Optional[str]
    
    # Self-correction loop
    retry_count: int
    max_retries: int
    correction_history: List[Dict[str, Any]]
    
    # Question decomposition (for complex queries)
    question_analysis: Optional[Dict[str, Any]]
    sub_questions: Optional[List[Dict[str, Any]]]
    sub_results: Optional[Dict[str, Any]]
    current_sub_index: int
    
    # RAG - Retrieved examples for few-shot prompting
    retrieved_examples: Optional[List[Dict[str, Any]]]
    
    # Output
    final_answer: Optional[str]
    should_terminate: bool


def create_initial_state(
    user_question: str,
    max_retries: int = 3,
    database_schema: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> AgentState:
    """
    Create an initial state for a new query.
    
    Args:
        user_question: The natural language question from the user
        max_retries: Maximum number of SQL correction attempts
        database_schema: Optional pre-loaded database schema
        conversation_context: Optional context from previous turns
        
    Returns:
        AgentState initialized with default values
    """
    from langchain_core.messages import HumanMessage
    
    return AgentState(
        messages=[HumanMessage(content=user_question)],
        user_question=user_question,
        normalized_question=None,
        conversation_context=conversation_context,
        database_schema=database_schema,
        table_metadata=None,
        relevant_tables=None,
        generated_sql=None,
        sql_validation_result=None,
        is_sql_valid=False,
        sql_result=None,
        execution_error=None,
        retry_count=0,
        max_retries=max_retries,
        correction_history=[],
        question_analysis=None,
        sub_questions=None,
        sub_results=None,
        current_sub_index=0,
        retrieved_examples=None,
        final_answer=None,
        should_terminate=False,
    )
