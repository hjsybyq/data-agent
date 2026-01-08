"""
Schema Query Tool

LangChain Tool for querying database schema information.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SchemaQueryInput(BaseModel):
    """Input schema for database schema query."""
    
    table_names: Optional[List[str]] = Field(
        default=None,
        description="List of specific table names to get schema for. If None, returns all tables."
    )


# In-memory schema storage (to be configured)
_schema_cache: Optional[str] = None


def set_database_schema(schema: str) -> None:
    """
    Set the database schema to be used by the tool.
    
    Args:
        schema: Full database schema as CREATE TABLE statements
    """
    global _schema_cache
    _schema_cache = schema


@tool("get_database_schema", args_schema=SchemaQueryInput)
def get_database_schema(table_names: Optional[List[str]] = None) -> str:
    """
    Get the database schema for specified tables or all tables.
    
    Args:
        table_names: Optional list of table names to filter
        
    Returns:
        Database schema as string
    """
    global _schema_cache
    
    if not _schema_cache:
        return "No database schema configured. Please set the schema first."
    
    if not table_names:
        return _schema_cache
    
    # Filter schema to specific tables
    import re
    
    filtered_parts = []
    for table in table_names:
        # Find CREATE TABLE statement for this table
        pattern = rf'CREATE TABLE\s+{re.escape(table)}\s*\([^)]+\);?'
        matches = re.findall(pattern, _schema_cache, re.IGNORECASE | re.DOTALL)
        filtered_parts.extend(matches)
    
    if filtered_parts:
        return "\n\n".join(filtered_parts)
    else:
        return f"No schema found for tables: {', '.join(table_names)}"
