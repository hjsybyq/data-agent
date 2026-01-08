"""
SQL Execution Tool

LangChain Tool for executing SQL queries against a database.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SQLExecutionInput(BaseModel):
    """Input schema for SQL execution."""
    
    sql: str = Field(description="SQL query to execute")


# Database connection (to be configured)  
_db_connection: Optional[Any] = None
_use_mock: bool = True


def set_database_connection(connection: Any) -> None:
    """
    Set the database connection to be used for SQL execution.
    
    Args:
        connection: Database connection object (SQLAlchemy engine, connection, etc.)
    """
    global _db_connection, _use_mock
    _db_connection = connection
    _use_mock = False


def enable_mock_mode() -> None:
    """Enable mock mode for testing."""
    global _use_mock
    _use_mock = True


@tool("execute_sql", args_schema=SQLExecutionInput)
def execute_sql(sql: str) -> Dict[str, Any]:
    """
    Execute a SQL query against the database.
    
    Args:
        sql: SQL query to execute
        
    Returns:
        Dictionary with success status, data, and metadata
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context, need to use different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, execute_sql_query(sql))
                return future.result()
        else:
            return loop.run_until_complete(execute_sql_query(sql))
    except RuntimeError:
        return asyncio.run(execute_sql_query(sql))


async def execute_sql_query(sql: str) -> Dict[str, Any]:
    """
    Async implementation of SQL execution.
    
    Args:
        sql: SQL query to execute
        
    Returns:
        Dictionary with success status, data, and metadata
    """
    global _db_connection, _use_mock
    
    if _use_mock or _db_connection is None:
        return _mock_execute(sql)
    
    try:
        import pandas as pd
        from sqlalchemy import text
        
        # Execute query
        if hasattr(_db_connection, 'execute'):
            # SQLAlchemy connection
            result = _db_connection.execute(text(sql))
            
            # Fetch results for SELECT queries
            if sql.strip().upper().startswith('SELECT'):
                rows = result.fetchall()
                columns = list(result.keys())
                
                # Convert to list of dicts
                data = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "columns": columns,
                }
            else:
                # For non-SELECT queries
                return {
                    "success": True,
                    "data": [],
                    "row_count": result.rowcount,
                    "columns": [],
                }
        else:
            # Try pandas read_sql
            df = pd.read_sql(sql, _db_connection)
            
            return {
                "success": True,
                "data": df.to_dict('records'),
                "row_count": len(df),
                "columns": df.columns.tolist(),
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None,
            "row_count": 0,
            "columns": [],
        }


def _mock_execute(sql: str) -> Dict[str, Any]:
    """
    Mock SQL execution for testing.
    
    Args:
        sql: SQL query
        
    Returns:
        Mock result dictionary
    """
    sql_upper = sql.upper().strip()
    
    # Check for syntax errors
    if not any(sql_upper.startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
        return {
            "success": False,
            "error": f"Invalid SQL syntax: query must start with SELECT, INSERT, UPDATE, DELETE, or WITH",
            "data": None,
            "row_count": 0,
            "columns": [],
        }
    
    # Return mock data for SELECT
    if sql_upper.startswith('SELECT') or sql_upper.startswith('WITH'):
        # Generate mock data based on query
        mock_data = [
            {"id": 1, "name": "Sample Result 1", "value": 100},
            {"id": 2, "name": "Sample Result 2", "value": 200},
            {"id": 3, "name": "Sample Result 3", "value": 300},
        ]
        
        return {
            "success": True,
            "data": mock_data,
            "row_count": 3,
            "columns": ["id", "name", "value"],
        }
    
    # For other queries
    return {
        "success": True,
        "data": [],
        "row_count": 1,
        "columns": [],
        "message": "Query executed successfully (mock mode)",
    }
