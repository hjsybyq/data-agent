"""
Result Evaluation Node

This node evaluates the SQL results and generates a natural language
response for the user. It also determines if the graph should terminate.
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage

from data_agent.graph.state import AgentState
from data_agent.utils.helpers import truncate_result


# Result summary prompt
RESULT_SUMMARY_PROMPT = """You are a helpful data analyst. Summarize the SQL query results in natural language for the user.

**IMPORTANT: Always respond in Chinese (中文).**

Original Question: {question}

SQL Query:
```sql
{sql}
```

Query Results:
{results}

请用中文提供清晰、简洁的总结，直接回答用户的问题。如果结果为空，请解释在该场景下的含义。包含数据中的关键数字和洞察。"""


async def result_evaluation(state: AgentState) -> Dict[str, Any]:
    """
    Evaluate SQL results and generate final response.
    
    This node:
    - Formats query results for presentation
    - Uses LLM to generate natural language summary
    - Sets termination flag to end the graph
    
    Args:
        state: Current graph state with sql_result
        
    Returns:
        State updates with final_answer and should_terminate
    """
    from data_agent.providers import get_llm
    
    question = state.get("normalized_question") or state["user_question"]
    sql = state.get("generated_sql", "")
    sql_result = state.get("sql_result")
    execution_error = state.get("execution_error")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # Handle execution error case
    if execution_error and retry_count >= max_retries:
        return {
            "final_answer": f"I was unable to answer your question after {max_retries} attempts. The last error was: {execution_error}",
            "should_terminate": True,
            "messages": [AIMessage(content=f"Query failed: {execution_error}")],
        }
    
    # Handle validation failure case
    if not sql_result and not execution_error:
        validation_result = state.get("sql_validation_result", {})
        errors = validation_result.get("errors", [])
        error_summary = "; ".join([e.get("message", "Unknown error") for e in errors])
        
        return {
            "final_answer": f"I was unable to generate a valid SQL query. Issues: {error_summary}",
            "should_terminate": True,
            "messages": [AIMessage(content=f"Validation failed: {error_summary}")],
        }
    
    # Format results for LLM
    if sql_result:
        data = sql_result.get("data", [])
        row_count = sql_result.get("row_count", len(data) if data else 0)
        columns = sql_result.get("columns", [])
        
        if data:
            # Truncate for LLM context
            truncated_data = truncate_result(data, max_rows=20)
            results_text = f"Columns: {', '.join(columns)}\n"
            results_text += f"Row count: {row_count}\n\n"
            results_text += "Sample data:\n"
            for i, row in enumerate(truncated_data[:10], 1):
                results_text += f"{i}. {row}\n"
            
            if row_count > 10:
                results_text += f"\n... and {row_count - 10} more rows"
        else:
            results_text = "No rows returned."
    else:
        results_text = "Query returned no data."
    
    # Generate summary using LLM
    prompt = RESULT_SUMMARY_PROMPT.format(
        question=question,
        sql=sql,
        results=results_text,
    )
    
    try:
        llm = get_llm()
        response = await llm.ainvoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        # Fallback to basic summary
        summary = f"Query executed successfully. {results_text}"
    
    return {
        "final_answer": summary,
        "should_terminate": True,
        "messages": [AIMessage(content=summary)],
    }
