"""
Sub-Query Executor Node

This node executes individual sub-queries as part of complex question
decomposition. It processes one sub-question at a time, respecting
dependencies between sub-questions.
"""

from typing import Dict, Any, List, Optional

from data_agent.graph.state import AgentState


async def sub_query_executor(state: AgentState) -> Dict[str, Any]:
    """
    Execute the current sub-query in the decomposition sequence.
    
    This node:
    1. Gets the current sub-question based on current_sub_index
    2. Generates SQL for this sub-question (using context from dependencies)
    3. Executes the SQL
    4. Stores result and advances to next sub-question
    
    Args:
        state: Current graph state with sub_questions and current_sub_index
        
    Returns:
        State updates with sub_results and incremented current_sub_index
    """
    from data_agent.providers import get_llm
    from data_agent.utils.helpers import extract_sql_from_response
    from data_agent.tools.sql_execution_tool import execute_sql_query
    
    sub_questions = state.get("sub_questions", [])
    current_index = state.get("current_sub_index", 0)
    sub_results = state.get("sub_results", {}) or {}
    schema = state.get("database_schema", "")
    
    if not sub_questions or current_index >= len(sub_questions):
        # No more sub-questions to process
        return {
            "current_sub_index": current_index,
        }
    
    # Get current sub-question
    current_sq = sub_questions[current_index]
    sq_id = current_sq.get("id", f"sq{current_index}")
    sq_question = current_sq.get("question", "")
    sq_depends_on = current_sq.get("depends_on", [])
    
    # Build context from dependencies
    dependency_context = _build_dependency_context(sq_depends_on, sub_results)
    
    # Generate SQL for this sub-question
    sql = await _generate_sub_query_sql(
        question=sq_question,
        schema=schema,
        dependency_context=dependency_context,
    )
    
    # Execute the SQL
    try:
        result = await execute_sql_query(sql)
        result_data = {
            "success": result.get("success", True),
            "data": result.get("data", []),
            "sql": sql,
        }
    except Exception as e:
        result_data = {
            "success": False,
            "error": str(e),
            "sql": sql,
        }
    
    # Update sub_results
    new_sub_results = {**sub_results, sq_id: result_data}
    
    # Update the sub_question with result
    updated_sub_questions = []
    for i, sq in enumerate(sub_questions):
        if i == current_index:
            updated_sq = {**sq, "sql": sql, "result": result_data, "completed": True}
            updated_sub_questions.append(updated_sq)
        else:
            updated_sub_questions.append(sq)
    
    return {
        "sub_questions": updated_sub_questions,
        "sub_results": new_sub_results,
        "current_sub_index": current_index + 1,
    }


def _build_dependency_context(depends_on: List[str], sub_results: Dict[str, Any]) -> str:
    """Build context string from dependency results."""
    if not depends_on or not sub_results:
        return ""
    
    context_parts = ["## Previous Query Results"]
    for dep_id in depends_on:
        if dep_id in sub_results:
            result = sub_results[dep_id]
            data = result.get("data", [])
            sql = result.get("sql", "")
            
            context_parts.append(f"\n### From query '{dep_id}':")
            context_parts.append(f"SQL: {sql}")
            if data:
                context_parts.append(f"Result: {str(data[:5])}...")
    
    return "\n".join(context_parts)


SUB_QUERY_PROMPT = """Generate SQL for the following sub-question.

## Database Schema
{schema}

{dependency_context}

## Sub-Question
{question}

Generate only the SQL query wrapped in ```sql ... ``` code blocks.
"""


async def _generate_sub_query_sql(
    question: str,
    schema: str,
    dependency_context: str,
) -> str:
    """Generate SQL for a single sub-question."""
    from data_agent.providers import get_llm
    from data_agent.utils.helpers import extract_sql_from_response
    
    prompt = SUB_QUERY_PROMPT.format(
        schema=schema[:3000] if schema else "No schema provided",
        dependency_context=dependency_context,
        question=question,
    )
    
    llm = get_llm()
    response = await llm.ainvoke(prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    return extract_sql_from_response(response_text)
