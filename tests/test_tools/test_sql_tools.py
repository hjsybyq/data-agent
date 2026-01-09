"""
Tests for SQL Tools

Tests for LangChain tool implementations.
"""

import pytest
from data_agent.tools.schema_tool import get_database_schema, set_database_schema
from data_agent.tools.sql_execution_tool import execute_sql, enable_mock_mode
from data_agent.tools.validation_tool import validate_sql


class TestSchemaTool:
    """Tests for schema query tool."""
    
    def test_get_all_schema(self, test_schema):
        """Test getting full schema."""
        set_database_schema(test_schema)
        result = get_database_schema.invoke({"table_names": None})
        
        assert "customers" in result.lower()
        assert "orders" in result.lower()
    
    def test_get_specific_tables(self, test_schema):
        """Test getting specific tables."""
        set_database_schema(test_schema)
        result = get_database_schema.invoke({"table_names": ["customers"]})
        
        assert "customers" in result.lower()
    
    def test_no_schema_configured(self):
        """Test behavior when no schema is configured."""
        set_database_schema("")
        result = get_database_schema.invoke({"table_names": None})
        
        # Should return empty or error message
        assert result is not None


class TestSqlExecutionTool:
    """Tests for SQL execution tool."""
    
    def test_execute_select(self):
        """Test executing a SELECT query."""
        enable_mock_mode()
        result = execute_sql.invoke({"sql": "SELECT * FROM customers"})
        
        assert result["success"] is True
        assert "data" in result
    
    def test_execute_invalid_sql(self):
        """Test executing invalid SQL."""
        enable_mock_mode()
        result = execute_sql.invoke({"sql": "INVALID SQL QUERY"})
        
        assert result["success"] is False


class TestValidationTool:
    """Tests for SQL validation tool."""
    
    def test_validate_valid_sql(self, test_schema):
        """Test validating valid SQL."""
        result = validate_sql.invoke({
            "sql": "SELECT * FROM customers",
            "schema_context": test_schema,
        })
        
        assert result["valid"] is True
    
    def test_validate_with_security_issue(self):
        """Test validation catches security issues."""
        result = validate_sql.invoke({
            "sql": "DROP TABLE users",
            "schema_context": "",
        })
        
        assert result["valid"] is False
        assert result["error_count"] > 0
