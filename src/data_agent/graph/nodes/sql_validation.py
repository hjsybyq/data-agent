"""
SQL Validation Node

This node validates the generated SQL query for syntax errors,
schema compatibility, and security concerns.
"""

from typing import Dict, Any, List
import re
import sqlparse

from data_agent.graph.state import AgentState


class SQLValidationError:
    """Represents a SQL validation error."""
    
    def __init__(self, error_type: str, message: str, severity: str = "error"):
        self.error_type = error_type
        self.message = message
        self.severity = severity
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "type": self.error_type,
            "message": self.message,
            "severity": self.severity,
        }


def validate_syntax(sql: str) -> List[SQLValidationError]:
    """
    Validate SQL syntax using sqlparse.
    
    Args:
        sql: SQL query string
        
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            errors.append(SQLValidationError(
                "syntax",
                "Failed to parse SQL query",
            ))
        elif len(parsed) > 1:
            errors.append(SQLValidationError(
                "syntax",
                "Multiple SQL statements detected. Please provide a single query.",
            ))
    except Exception as e:
        errors.append(SQLValidationError(
            "syntax",
            f"SQL parsing error: {str(e)}",
        ))
    
    return errors


def validate_schema_compatibility(
    sql: str, 
    schema: str,
    relevant_tables: List[str],
) -> List[SQLValidationError]:
    """
    Check if SQL references valid tables from the schema.
    
    Args:
        sql: SQL query string
        schema: Database schema
        relevant_tables: List of valid table names
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Extract tables from schema
    schema_tables = set()
    table_pattern = r'CREATE TABLE\s+(\w+)\s*\('
    for match in re.finditer(table_pattern, schema, re.IGNORECASE):
        schema_tables.add(match.group(1).lower())
    
    # Extract table references from SQL (basic extraction)
    sql_lower = sql.lower()
    
    # Look for FROM and JOIN clauses
    table_ref_pattern = r'(?:from|join)\s+(\w+)'
    referenced_tables = set()
    for match in re.finditer(table_ref_pattern, sql_lower):
        referenced_tables.add(match.group(1))
    
    # Check for unknown tables
    for table in referenced_tables:
        if table not in schema_tables:
            errors.append(SQLValidationError(
                "schema",
                f"Table '{table}' not found in database schema",
            ))
    
    return errors


def validate_security(sql: str) -> List[SQLValidationError]:
    """
    Check for potential security issues in SQL.
    
    Args:
        sql: SQL query string
        
    Returns:
        List of validation errors
    """
    errors = []
    sql_upper = sql.upper()
    
    # Dangerous statements
    dangerous_patterns = [
        (r'\bDROP\s+(TABLE|DATABASE|INDEX)', "DROP statements are not allowed"),
        (r'\bTRUNCATE\s+TABLE', "TRUNCATE statements are not allowed"),
        (r'\bDELETE\s+FROM\s+\w+\s*(WHERE\s+1\s*=\s*1|$)', "Unrestricted DELETE is not allowed"),
        (r'\bUPDATE\s+\w+\s+SET\s+.*(WHERE\s+1\s*=\s*1|$)', "Unrestricted UPDATE is not allowed"),
        (r';\s*(DROP|DELETE|UPDATE|INSERT)', "Multiple statements are not allowed"),
        (r'--', "SQL comments may indicate injection attempt"),
        (r'/\*', "Block comments may indicate injection attempt"),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, sql_upper):
            errors.append(SQLValidationError(
                "security",
                message,
                severity="warning" if "comment" in message.lower() else "error",
            ))
    
    return errors


async def sql_validation(state: AgentState) -> Dict[str, Any]:
    """
    Validate the generated SQL query.
    
    This node:
    - Checks SQL syntax
    - Verifies schema compatibility  
    - Performs security checks
    - Updates state with validation results
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with validation results
    """
    sql = state.get("generated_sql")
    schema = state.get("database_schema", "")
    relevant_tables = state.get("relevant_tables", [])
    correction_history = state.get("correction_history", [])
    
    if not sql:
        return {
            "is_sql_valid": False,
            "sql_validation_result": {
                "valid": False,
                "errors": [{"type": "missing", "message": "No SQL query generated"}],
            },
        }
    
    # Run all validations
    all_errors = []
    all_errors.extend(validate_syntax(sql))
    all_errors.extend(validate_schema_compatibility(sql, schema, relevant_tables))
    all_errors.extend(validate_security(sql))
    
    # Filter to only blocking errors (not warnings)
    blocking_errors = [e for e in all_errors if e.severity == "error"]
    
    is_valid = len(blocking_errors) == 0
    
    # Build validation result
    validation_result = {
        "valid": is_valid,
        "errors": [e.to_dict() for e in all_errors],
        "blocking_error_count": len(blocking_errors),
    }
    
    # If invalid, add to correction history
    new_correction_history = correction_history.copy()
    if not is_valid and blocking_errors:
        error_messages = "; ".join([e.message for e in blocking_errors])
        new_correction_history.append({
            "original_sql": sql,
            "error": error_messages,
            "attempt_number": len(correction_history) + 1,
        })
    
    return {
        "is_sql_valid": is_valid,
        "sql_validation_result": validation_result,
        "correction_history": new_correction_history,
    }
