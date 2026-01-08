"""
Schema Acquisition Node

This node retrieves and prepares database schema information
relevant to the user's question.
"""

from typing import Dict, Any, Optional, List
import re

from data_agent.graph.state import AgentState


# Default schema for demo/testing - will be replaced by actual DB schema
DEFAULT_SCHEMA = """
-- Sample Schema (replace with actual database schema)

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20)
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock_quantity INTEGER
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER,
    unit_price DECIMAL(10, 2)
);
"""


def extract_table_hints(question: str) -> List[str]:
    """
    Extract potential table names mentioned in the question.
    
    Args:
        question: The user's question
        
    Returns:
        List of potential table names
    """
    # Common table name patterns
    hints = []
    
    # Look for common entity references
    entity_patterns = {
        r'\bcustomer': 'customers',
        r'\border': 'orders',
        r'\bproduct': 'products',
        r'\bitem': 'order_items',
        r'\bsale': 'orders',
        r'\buser': 'customers',
        r'\bpurchase': 'orders',
    }
    
    question_lower = question.lower()
    for pattern, table in entity_patterns.items():
        if re.search(pattern, question_lower):
            if table not in hints:
                hints.append(table)
    
    return hints


async def schema_acquisition(state: AgentState) -> Dict[str, Any]:
    """
    Acquire relevant database schema for SQL generation.
    
    This node:
    - Retrieves database schema (from state or default)
    - Identifies tables relevant to the question
    - Prepares metadata for SQL generation
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with schema information
    """
    question = state.get("normalized_question") or state["user_question"]
    
    # Use provided schema or default
    schema = state.get("database_schema") or DEFAULT_SCHEMA
    
    # Extract table hints from question
    relevant_tables = extract_table_hints(question)
    
    # Build basic table metadata
    table_metadata = {}
    
    # Parse schema to extract table info (basic parsing)
    table_pattern = r'CREATE TABLE\s+(\w+)\s*\('
    for match in re.finditer(table_pattern, schema, re.IGNORECASE):
        table_name = match.group(1)
        table_metadata[table_name] = {
            "mentioned_in_question": table_name in relevant_tables,
        }
    
    # If no tables detected, include all tables
    if not relevant_tables:
        relevant_tables = list(table_metadata.keys())
    
    return {
        "database_schema": schema,
        "relevant_tables": relevant_tables,
        "table_metadata": table_metadata,
    }
