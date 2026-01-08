"""
Tests for VannaState and State Factory

Tests the state schema definitions and factory functions.
"""

import pytest
from vanna_langgraph.graph.state import (
    VannaState,
    SQLResult,
    CorrectionEntry,
    create_initial_state,
)


class TestSQLResult:
    """Tests for SQLResult model."""
    
    def test_successful_result(self):
        """Test creating a successful result."""
        result = SQLResult(
            success=True,
            data=[{"id": 1, "name": "Test"}],
            row_count=1,
            columns=["id", "name"],
        )
        
        assert result.success is True
        assert result.row_count == 1
        assert len(result.data) == 1
        assert result.error is None
    
    def test_failed_result(self):
        """Test creating a failed result."""
        result = SQLResult(
            success=False,
            error="Syntax error near SELECT",
        )
        
        assert result.success is False
        assert result.error == "Syntax error near SELECT"
        assert result.data is None
    
    def test_empty_result(self):
        """Test creating an empty result."""
        result = SQLResult(success=True, data=[], row_count=0, columns=["id"])
        
        assert result.success is True
        assert result.row_count == 0
        assert result.data == []


class TestCorrectionEntry:
    """Tests for CorrectionEntry model."""
    
    def test_correction_entry(self):
        """Test creating a correction entry."""
        entry = CorrectionEntry(
            original_sql="SELECT * FORM users",
            error="Syntax error: FORM should be FROM",
            corrected_sql="SELECT * FROM users",
            attempt_number=1,
        )
        
        assert entry.original_sql == "SELECT * FORM users"
        assert entry.attempt_number == 1
        assert entry.corrected_sql == "SELECT * FROM users"


class TestCreateInitialState:
    """Tests for create_initial_state factory."""
    
    def test_basic_initial_state(self):
        """Test creating basic initial state."""
        state = create_initial_state("How many users?")
        
        assert state["user_question"] == "How many users?"
        assert state["normalized_question"] is None
        assert state["retry_count"] == 0
        assert state["max_retries"] == 3
        assert state["is_sql_valid"] is False
        assert state["should_terminate"] is False
        assert len(state["messages"]) == 1
    
    def test_initial_state_with_schema(self):
        """Test creating state with schema."""
        schema = "CREATE TABLE users (id INT);"
        state = create_initial_state("Count users", database_schema=schema)
        
        assert state["database_schema"] == schema
    
    def test_initial_state_custom_retries(self):
        """Test creating state with custom max retries."""
        state = create_initial_state("Query", max_retries=5)
        
        assert state["max_retries"] == 5
    
    def test_initial_state_messages(self):
        """Test that initial state has human message."""
        from langchain_core.messages import HumanMessage
        
        state = create_initial_state("Test question")
        
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "Test question"


class TestVannaStateTyping:
    """Tests for VannaState type structure."""
    
    def test_state_has_required_fields(self):
        """Test that state has all required fields."""
        state = create_initial_state("Test")
        
        required_fields = [
            "messages",
            "user_question",
            "normalized_question",
            "database_schema",
            "table_metadata",
            "relevant_tables",
            "generated_sql",
            "sql_validation_result",
            "is_sql_valid",
            "sql_result",
            "execution_error",
            "retry_count",
            "max_retries",
            "correction_history",
            "final_answer",
            "should_terminate",
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"
