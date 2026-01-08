"""
SQL Generation Node

This node generates SQL queries from natural language using an LLM
with schema context and correction history.
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage

from data_agent.graph.state import AgentState
from data_agent.utils.helpers import (
    extract_sql_from_response,
    build_schema_context,
    create_correction_context,
)


def _format_examples_context(examples: list) -> str:
    """
    Format retrieved examples for inclusion in the SQL generation prompt.
    
    Args:
        examples: List of retrieved example dicts with question, sql, score
        
    Returns:
        Formatted string for prompt, or empty string if no examples
    """
    if not examples:
        return ""
    
    parts = ["## Similar Examples\n"]
    parts.append("Use these similar queries as reference for style and patterns:\n\n")
    
    for i, ex in enumerate(examples, 1):
        parts.append(f"**Example {i}:**\n")
        parts.append(f"Question: {ex.get('question', '')}\n")
        parts.append(f"SQL:\n```sql\n{ex.get('sql', '')}\n```\n\n")
    
    return "".join(parts)


# SQL generation system prompt
SQL_GENERATION_PROMPT = """You are an expert SQL query generator. Given a natural language question and database schema, generate a SQL query that answers the question.

Rules:
1. Generate only valid SQL syntax
2. Use only tables and columns that exist in the provided schema
3. Return ONLY the SQL query wrapped in ```sql ... ``` code blocks
4. Use appropriate JOINs when data spans multiple tables
5. Handle NULL values appropriately
6. Use aliases for readability when joining tables
7. Include appropriate WHERE clauses for filtering
8. Use aggregate functions (COUNT, SUM, AVG, etc.) when needed
9. Add ORDER BY for sorted results when implied by the question
10. Use LIMIT when the question asks for "top N" or similar
11. Pay attention to conversation context - resolve pronouns and references based on previous exchanges
12. Learn from the similar examples provided below - follow their patterns and style

{examples_context}

{conversation_context}

{schema_context}

{correction_context}

Question: {question}

Generate the SQL query:"""


async def sql_generation(state: AgentState) -> Dict[str, Any]:
    """
    Generate SQL from the user's question using LLM.
    
    This node:
    - Builds context from schema, conversation, and correction history
    - Calls LLM to generate SQL
    - Extracts SQL from the LLM response
    - Increments retry count if this is a correction attempt
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with generated_sql and updated messages
    """
    from data_agent.providers import get_llm
    
    question = state.get("normalized_question") or state["user_question"]
    schema = state.get("database_schema", "")
    relevant_tables = state.get("relevant_tables", [])
    table_metadata = state.get("table_metadata", {})
    correction_history = state.get("correction_history", [])
    conversation_context = state.get("conversation_context", "") or ""
    retry_count = state.get("retry_count", 0)
    
    # Build context
    schema_context = build_schema_context(schema, relevant_tables, table_metadata)
    correction_context = create_correction_context(correction_history)
    
    # Build examples context from retrieved examples
    retrieved_examples = state.get("retrieved_examples", []) or []
    examples_context = _format_examples_context(retrieved_examples)
    
    # Build the prompt with all context
    prompt = SQL_GENERATION_PROMPT.format(
        examples_context=examples_context,
        conversation_context=conversation_context,
        schema_context=schema_context,
        correction_context=correction_context,
        question=question,
    )
    
    # Get LLM and generate
    llm = get_llm()
    response = await llm.ainvoke(prompt)
    
    # Extract response content
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Extract SQL from response
    generated_sql = extract_sql_from_response(response_text)
    
    # Create AI message for conversation tracking
    ai_message = AIMessage(content=response_text)
    
    # Determine if this is a retry
    is_retry = len(correction_history) > 0
    new_retry_count = retry_count + 1 if is_retry else retry_count
    
    return {
        "generated_sql": generated_sql,
        "messages": [ai_message],
        "retry_count": new_retry_count,
        # Reset validation status for new SQL
        "is_sql_valid": False,
        "sql_validation_result": None,
        "execution_error": None,
    }
