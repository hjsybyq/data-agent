"""
SQL Validation Tool

LangChain Tool for validating SQL queries.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SQLValidationInput(BaseModel):
    """Input schema for SQL validation."""
    
    sql: str = Field(description="SQL query to validate")
    schema_context: str = Field(
        default="", 
        description="Database schema context for validation"
    )


@tool("validate_sql", args_schema=SQLValidationInput)
def validate_sql(sql: str, schema_context: str = "") -> Dict[str, Any]:
    """
    Validate a SQL query for syntax and schema compatibility.
    
    Args:
        sql: SQL query to validate
        schema_context: Database schema for compatibility checking
        
    Returns:
        Dictionary with validation results
    """
    from data_agent.graph.nodes.sql_validation import (
        validate_syntax,
        validate_schema_compatibility,
        validate_security,
    )
    
    all_errors = []
    
    # Run validations
    all_errors.extend(validate_syntax(sql))
    
    if schema_context:
        all_errors.extend(validate_schema_compatibility(sql, schema_context, []))
    
    all_errors.extend(validate_security(sql))
    
    # Separate blocking errors from warnings
    blocking_errors = [e for e in all_errors if e.severity == "error"]
    warnings = [e for e in all_errors if e.severity == "warning"]
    
    return {
        "valid": len(blocking_errors) == 0,
        "errors": [e.to_dict() for e in blocking_errors],
        "warnings": [e.to_dict() for e in warnings],
        "error_count": len(blocking_errors),
        "warning_count": len(warnings),
    }
