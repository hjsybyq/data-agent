"""
Tests for Vanna Adapter

Tests the compatibility layer with original Vanna API.
"""

import pytest
from vanna_langgraph.adapters.vanna_adapter import VannaLangGraph, VannaConfig


class TestVannaConfig:
    """Tests for VannaConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VannaConfig()
        
        assert config.llm_provider == "openai"
        assert config.max_retries == 3
        assert config.temperature == 0.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = VannaConfig(
            llm_provider="anthropic",
            max_retries=5,
            temperature=0.5,
        )
        
        assert config.llm_provider == "anthropic"
        assert config.max_retries == 5


class TestVannaLangGraphInit:
    """Tests for VannaLangGraph initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        vn = VannaLangGraph()
        
        assert vn.config.llm_provider == "openai"
        assert vn._graph is not None
    
    def test_custom_provider(self):
        """Test initialization with custom provider."""
        vn = VannaLangGraph(llm_provider="anthropic")
        
        assert vn.config.llm_provider == "anthropic"
    
    def test_simple_graph_option(self):
        """Test simple graph option."""
        vn = VannaLangGraph(use_simple_graph=True)
        
        assert vn.config.use_simple_graph is True


class TestVannaLangGraphSchema:
    """Tests for schema management."""
    
    def test_set_schema(self, test_schema):
        """Test setting database schema."""
        vn = VannaLangGraph()
        vn.set_schema(test_schema)
        
        assert vn._schema == test_schema
        assert vn.config.database_schema == test_schema
    
    def test_get_related_ddl(self, test_schema):
        """Test getting related DDL."""
        vn = VannaLangGraph()
        vn.set_schema(test_schema)
        
        result = vn.get_related_ddl("any question")
        
        assert result == test_schema


class TestVannaLangGraphMethods:
    """Tests for API methods."""
    
    def test_generate_sql_method_exists(self):
        """Test that generate_sql method exists."""
        vn = VannaLangGraph()
        
        assert hasattr(vn, "generate_sql")
        assert callable(vn.generate_sql)
    
    def test_run_sql_method_exists(self):
        """Test that run_sql method exists."""
        vn = VannaLangGraph()
        
        assert hasattr(vn, "run_sql")
        assert callable(vn.run_sql)
    
    def test_ask_method_exists(self):
        """Test that ask method exists."""
        vn = VannaLangGraph()
        
        assert hasattr(vn, "ask")
        assert callable(vn.ask)
    
    def test_train_method_exists(self):
        """Test that train method exists (compatibility)."""
        vn = VannaLangGraph()
        
        assert hasattr(vn, "train")
        assert callable(vn.train)
    
    def test_train_with_ddl(self, test_schema):
        """Test train with DDL sets schema."""
        vn = VannaLangGraph()
        vn.train(ddl=test_schema)
        
        assert vn._schema == test_schema
    
    def test_graph_property(self):
        """Test graph property exposes underlying graph."""
        vn = VannaLangGraph()
        
        assert vn.graph is not None
