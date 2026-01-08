# Data Agent

A LangGraph-based Text-to-SQL system. This project provides a graph-driven approach to natural language SQL generation with explicit state management, retry loops, and LangChain integration.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- 🔄 **Graph-Driven Architecture** - Explicit flow control with LangGraph StateGraph
- 🛠️ **Self-Correcting SQL** - Automatic retry loops for SQL validation and execution errors
- 🔌 **Provider Agnostic** - Support for OpenAI, Anthropic, and Google LLMs
- 📚 **RAG Example Retrieval** - Learn from question-SQL examples for better accuracy
- 🔀 **Question Decomposition** - Automatically break down complex queries
- 🗣️ **Multi-turn Conversations** - Context-aware follow-up questions

## Quick Start

```bash
# Install
pip install -e ".[openai]"

# Set your API key
export OPENAI_API_KEY="your-key"
```

```python
from data_agent import DataAgent

# Initialize
vn = DataAgent(llm_provider="openai")

# Set your database schema
vn.set_schema("""
    CREATE TABLE customers (id INT, name VARCHAR(100));
    CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL);
""")

# Ask questions
result = vn.ask("How many customers do we have?")
print(result["sql"])     # SELECT COUNT(*) FROM customers
print(result["answer"])  # Natural language response
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Input       │───▶│     Schema      │───▶│    Example      │
│  Normalization  │    │   Acquisition   │    │   Retrieval     │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                       ┌───────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │      SQL        │──── Invalid ───┐
              │   Validation    │                │
              └────────┬────────┘                │
                       │ Valid                   │ Retry
                       ▼                         │
              ┌─────────────────┐                │
              │      SQL        │──── Error ─────┘
              │   Execution     │
              └────────┬────────┘
                       │ Success
                       ▼
              ┌─────────────────┐
              │     Result      │
              │   Evaluation    │
              └─────────────────┘
```

## Installation

```bash
# Base installation
pip install -e .

# With specific LLM providers
pip install -e ".[openai]"      # OpenAI
pip install -e ".[anthropic]"   # Anthropic
pip install -e ".[google]"      # Google GenAI
pip install -e ".[all-providers]"  # All providers

# Development
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from data_agent import DataAgent

vn = DataAgent()
vn.set_schema("CREATE TABLE users (id INT, name TEXT);")

# Generate SQL only
sql = vn.generate_sql("Count all users")

# Execute SQL
result = vn.run_sql("SELECT COUNT(*) FROM users")

# Full pipeline
response = vn.ask("How many users?")
```

### OpenAI Compatible API

```python
vn = DataAgent(
    llm_provider="openai_compatible",
    base_url="https://api.siliconflow.cn/v1",
    llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    llm_api_key="your-api-key",
    embedding_model="BAAI/bge-m3",  # Custom embedding model
)
```

### RAG Training

```python
# Train with examples
vn.train(question="How many customers?", sql="SELECT COUNT(*) FROM Customer")
vn.train(question="Top selling products", sql="SELECT * FROM Product ORDER BY Sales DESC LIMIT 10")

# View training data
examples = vn.get_training_data()

# Search similar examples
similar = vn.search_similar_examples("customer count", k=3)
```

### Connect to Real Database

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/db")
vn = DataAgent(database_connection=engine.connect())
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=src/data_agent
```

## Project Structure

```
src/data_agent/
├── graph/              # LangGraph core
│   ├── state.py       # AgentState schema
│   ├── builder.py     # Graph construction
│   ├── edges.py       # Conditional edge functions
│   └── nodes/         # Node implementations
├── rag/               # RAG example retrieval
├── tools/             # LangChain tools
├── providers/         # LLM provider abstraction
├── adapters/          # Compatibility layer
└── utils/             # Helper functions
```

## License

MIT License - See [LICENSE](LICENSE) for details.
