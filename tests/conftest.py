"""
Pytest Configuration and Fixtures

Common fixtures for testing Data Agent components.
"""

import pytest
from typing import Dict, Any

from data_agent.graph.state import AgentState, create_initial_state
from data_agent.tools.sql_execution_tool import enable_mock_mode
from data_agent.tools.schema_tool import set_database_schema


# Sample test schema
TEST_SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);
"""


@pytest.fixture(autouse=True)
def setup_mock_mode():
    """Enable mock mode for all tests by default."""
    enable_mock_mode()
    set_database_schema(TEST_SCHEMA)
    yield


@pytest.fixture
def test_schema() -> str:
    """Provide the test schema."""
    return TEST_SCHEMA


@pytest.fixture
def sample_question() -> str:
    """Provide a sample question for testing."""
    return "How many customers do we have?"


@pytest.fixture
def initial_state(sample_question: str, test_schema: str) -> AgentState:
    """Create an initial state for testing."""
    return create_initial_state(
        user_question=sample_question,
        max_retries=3,
        database_schema=test_schema,
    )


@pytest.fixture
def complex_question() -> str:
    """Provide a complex question requiring joins."""
    return "Show me the top 5 customers by total order amount"


@pytest.fixture
def sample_sql() -> str:
    """Provide a sample SQL query."""
    return "SELECT COUNT(*) as customer_count FROM customers"


@pytest.fixture
def sample_result() -> Dict[str, Any]:
    """Provide a sample SQL result."""
    return {
        "success": True,
        "data": [{"customer_count": 100}],
        "row_count": 1,
        "columns": ["customer_count"],
    }
