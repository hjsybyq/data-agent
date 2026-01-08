"""
Tests for Graph Nodes

Tests individual node functions in isolation.
"""

import pytest
from vanna_langgraph.graph.state import create_initial_state


class TestInputNormalization:
    """Tests for input normalization node."""
    
    @pytest.mark.asyncio
    async def test_basic_normalization(self, initial_state):
        """Test basic input normalization."""
        from vanna_langgraph.graph.nodes.input_normalization import input_normalization
        
        result = await input_normalization(initial_state)
        
        assert "normalized_question" in result
        assert result["normalized_question"] is not None
    
    @pytest.mark.asyncio
    async def test_whitespace_cleanup(self):
        """Test that whitespace is cleaned up."""
        from vanna_langgraph.graph.nodes.input_normalization import input_normalization
        
        state = create_initial_state("  How   many   customers  ")
        result = await input_normalization(state)
        
        # Should have single spaces and proper ending
        assert "  " not in result["normalized_question"]
    
    @pytest.mark.asyncio
    async def test_abbreviation_expansion(self):
        """Test that abbreviations are expanded."""
        from vanna_langgraph.graph.nodes.input_normalization import input_normalization
        
        state = create_initial_state("What is the avg order amount?")
        result = await input_normalization(state)
        
        assert "average" in result["normalized_question"].lower()
    
    @pytest.mark.asyncio
    async def test_adds_question_mark(self):
        """Test that question mark is added if missing."""
        from vanna_langgraph.graph.nodes.input_normalization import input_normalization
        
        state = create_initial_state("How many customers")
        result = await input_normalization(state)
        
        assert result["normalized_question"].endswith("?")


class TestSchemaAcquisition:
    """Tests for schema acquisition node."""
    
    @pytest.mark.asyncio
    async def test_schema_acquisition(self, initial_state, test_schema):
        """Test schema acquisition."""
        from vanna_langgraph.graph.nodes.schema_acquisition import schema_acquisition
        
        result = await schema_acquisition(initial_state)
        
        assert "database_schema" in result
        assert "relevant_tables" in result
        assert "table_metadata" in result
    
    @pytest.mark.asyncio
    async def test_table_detection(self, test_schema):
        """Test that relevant tables are detected."""
        from vanna_langgraph.graph.nodes.schema_acquisition import schema_acquisition
        
        state = create_initial_state(
            "How many orders were placed?",
            database_schema=test_schema,
        )
        result = await schema_acquisition(state)
        
        assert "orders" in result["relevant_tables"]
    
    @pytest.mark.asyncio
    async def test_multiple_table_detection(self, test_schema):
        """Test detection of multiple tables."""
        from vanna_langgraph.graph.nodes.schema_acquisition import schema_acquisition
        
        state = create_initial_state(
            "Show customer orders",
            database_schema=test_schema,
        )
        result = await schema_acquisition(state)
        
        assert "customers" in result["relevant_tables"]
        assert "orders" in result["relevant_tables"]


class TestSqlValidation:
    """Tests for SQL validation node."""
    
    @pytest.mark.asyncio
    async def test_valid_sql(self, test_schema):
        """Test validation of valid SQL."""
        from vanna_langgraph.graph.nodes.sql_validation import sql_validation
        
        state = create_initial_state("Count customers", database_schema=test_schema)
        state["generated_sql"] = "SELECT COUNT(*) FROM customers"
        
        result = await sql_validation(state)
        
        assert result["is_sql_valid"] is True
    
    @pytest.mark.asyncio
    async def test_invalid_table(self, test_schema):
        """Test validation catches invalid table."""
        from vanna_langgraph.graph.nodes.sql_validation import sql_validation
        
        state = create_initial_state("Query", database_schema=test_schema)
        state["generated_sql"] = "SELECT * FROM nonexistent_table"
        state["correction_history"] = []
        
        result = await sql_validation(state)
        
        assert result["is_sql_valid"] is False
        assert len(result["correction_history"]) > 0
    
    @pytest.mark.asyncio
    async def test_security_check(self, test_schema):
        """Test security validation."""
        from vanna_langgraph.graph.nodes.sql_validation import sql_validation
        
        state = create_initial_state("Delete", database_schema=test_schema)
        state["generated_sql"] = "DROP TABLE customers"
        state["correction_history"] = []
        
        result = await sql_validation(state)
        
        assert result["is_sql_valid"] is False
    
    @pytest.mark.asyncio
    async def test_missing_sql(self, initial_state):
        """Test validation with no SQL."""
        from vanna_langgraph.graph.nodes.sql_validation import sql_validation
        
        result = await sql_validation(initial_state)
        
        assert result["is_sql_valid"] is False


class TestSqlExecution:
    """Tests for SQL execution node."""
    
    @pytest.mark.asyncio
    async def test_execution_success(self, test_schema):
        """Test successful SQL execution."""
        from vanna_langgraph.graph.nodes.sql_execution import sql_execution
        
        state = create_initial_state("Count", database_schema=test_schema)
        state["generated_sql"] = "SELECT COUNT(*) FROM customers"
        state["correction_history"] = []
        
        result = await sql_execution(state)
        
        assert result.get("sql_result") is not None
        assert result.get("execution_error") is None
    
    @pytest.mark.asyncio
    async def test_execution_no_sql(self, initial_state):
        """Test execution with no SQL."""
        from vanna_langgraph.graph.nodes.sql_execution import sql_execution
        
        result = await sql_execution(initial_state)
        
        assert result["execution_error"] is not None
        assert result["sql_result"] is None
