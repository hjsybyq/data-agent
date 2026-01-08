"""
Helper Utilities for Vanna LangGraph

Common helper functions used across the codebase.
"""

import re
from typing import Any, Dict, List, Optional
import sqlparse


def format_sql(sql: str) -> str:
    """
    Format SQL query for better readability.
    
    Args:
        sql: Raw SQL query string
        
    Returns:
        Formatted SQL string
    """
    try:
        return sqlparse.format(
            sql,
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
        )
    except Exception:
        # Return original if formatting fails
        return sql.strip()


def truncate_result(
    data: List[Dict[str, Any]], 
    max_rows: int = 100,
    max_str_length: int = 500,
) -> List[Dict[str, Any]]:
    """
    Truncate result data for display and LLM context.
    
    Args:
        data: List of row dictionaries
        max_rows: Maximum number of rows to include
        max_str_length: Maximum length for string values
        
    Returns:
        Truncated data list
    """
    truncated = data[:max_rows]
    
    # Truncate long string values
    for row in truncated:
        for key, value in row.items():
            if isinstance(value, str) and len(value) > max_str_length:
                row[key] = value[:max_str_length] + "..."
                
    return truncated


def extract_sql_from_response(response: str) -> Optional[str]:
    """
    Extract SQL query from LLM response text.
    
    Handles various formats:
    - SQL in code blocks (```sql ... ```)
    - SQL in generic code blocks (``` ... ```)
    - Raw SQL statements
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted SQL or None if not found
    """
    # Try to extract from SQL code block
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to extract from generic code block
    code_block_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        potential_sql = match.group(1).strip()
        # Check if it looks like SQL
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
        if any(potential_sql.upper().startswith(kw) for kw in sql_keywords):
            return potential_sql
    
    # Try to find raw SQL statement
    for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']:
        pattern = rf'\b{keyword}\b.*?(?:;|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(0).strip()
            # Remove trailing semicolon for consistency
            return sql.rstrip(';').strip()
    
    return None


def build_schema_context(
    schema: str,
    relevant_tables: Optional[List[str]] = None,
    table_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build schema context string for LLM prompts.
    
    Args:
        schema: Full database schema
        relevant_tables: List of relevant table names
        table_metadata: Additional metadata about tables
        
    Returns:
        Formatted schema context string
    """
    parts = ["## Database Schema\n"]
    
    if relevant_tables:
        parts.append(f"Relevant tables: {', '.join(relevant_tables)}\n\n")
    
    parts.append(schema)
    
    if table_metadata:
        parts.append("\n\n## Table Metadata\n")
        for table, metadata in table_metadata.items():
            parts.append(f"\n### {table}\n")
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    parts.append(f"- {key}: {value}\n")
            else:
                parts.append(f"{metadata}\n")
    
    return "".join(parts)


def create_correction_context(
    correction_history: List[Dict[str, Any]],
) -> str:
    """
    Build context from previous correction attempts for the LLM.
    
    Args:
        correction_history: List of previous correction attempts
        
    Returns:
        Formatted correction context string
    """
    if not correction_history:
        return ""
    
    parts = ["## Previous Attempts\n\n"]
    parts.append("The following SQL queries were tried but failed. ")
    parts.append("Learn from these errors:\n\n")
    
    for i, entry in enumerate(correction_history, 1):
        parts.append(f"### Attempt {i}\n")
        parts.append(f"**SQL:**\n```sql\n{entry.get('original_sql', 'N/A')}\n```\n")
        parts.append(f"**Error:** {entry.get('error', 'Unknown error')}\n\n")
    
    return "".join(parts)
