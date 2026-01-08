"""
SQL Execution Node

This node executes the validated SQL query against the database
and captures results or errors.
"""

from typing import Dict, Any

from data_agent.graph.state import AgentState


async def sql_execution(state: AgentState) -> Dict[str, Any]:
    """
    Execute the validated SQL query.
    
    This node:
    - Executes SQL using the configured database connector
    - Captures results as structured data
    - Handles execution errors gracefully
    - Updates correction history on failure
    
    Args:
        state: Current graph state with validated SQL
        
    Returns:
        State updates with sql_result or execution_error
    """
    sql = state.get("generated_sql")
    correction_history = state.get("correction_history", [])
    
    if not sql:
        return {
            "execution_error": "No SQL query to execute",
            "sql_result": None,
        }
    
    try:
        # Import the execution tool
        from data_agent.tools.sql_execution_tool import execute_sql_query
        
        # Execute the query
        result = await execute_sql_query(sql)
        
        if result.get("success", False):
            return {
                "sql_result": result,
                "execution_error": None,
            }
        else:
            error_message = result.get("error", "Unknown execution error")
            
            # Add to correction history
            new_correction_history = correction_history.copy()
            new_correction_history.append({
                "original_sql": sql,
                "error": error_message,
                "attempt_number": len(correction_history) + 1,
            })
            
            return {
                "sql_result": None,
                "execution_error": error_message,
                "correction_history": new_correction_history,
            }
            
    except Exception as e:
        error_message = f"Execution failed: {str(e)}"
        
        # Add to correction history
        new_correction_history = correction_history.copy()
        new_correction_history.append({
            "original_sql": sql,
            "error": error_message,
            "attempt_number": len(correction_history) + 1,
        })
        
        return {
            "sql_result": None,
            "execution_error": error_message,
            "correction_history": new_correction_history,
        }
