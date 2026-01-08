# Vanna LangGraph

A LangGraph-based refactoring of the [Vanna](https://github.com/vanna-ai/vanna) text-to-SQL system. This project provides a graph-driven approach to natural language SQL generation with explicit state management, retry loops, and LangChain integration.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- 🔄 **Graph-Driven Architecture** - Explicit flow control with LangGraph StateGraph
- 🛠️ **Self-Correcting SQL** - Automatic retry loops for SQL validation and execution errors
- 🔌 **Provider Agnostic** - Support for OpenAI, Anthropic, and Google LLMs
- 🧪 **Fully Tested** - Comprehensive test suite with pytest
- 🔙 **API Compatible** - Familiar interface for Vanna users

## Quick Start

```bash
# Install
pip install -e ".[openai]"

# Set your API key
export OPENAI_API_KEY="your-key"
```

```python
from vanna_langgraph import VannaLangGraph

# Initialize
vn = VannaLangGraph(llm_provider="openai")

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
│     Input       │───▶│     Schema      │───▶│      SQL        │
│  Normalization  │    │   Acquisition   │    │   Generation    │
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
from vanna_langgraph import VannaLangGraph

vn = VannaLangGraph()
vn.set_schema("CREATE TABLE users (id INT, name TEXT);")

# Generate SQL only
sql = vn.generate_sql("Count all users")

# Execute SQL
result = vn.run_sql("SELECT COUNT(*) FROM users")

# Full pipeline
response = vn.ask("How many users?")
```

### Custom LLM Provider

```python
vn = VannaLangGraph(
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514",
    llm_api_key="your-anthropic-key",
)
```

### Connect to Real Database

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/db")
vn = VannaLangGraph(database_connection=engine.connect())
```

### Access Underlying Graph

```python
# Get the LangGraph for advanced usage
graph = vn.graph

# Invoke with custom state
from vanna_langgraph.graph.state import create_initial_state
state = create_initial_state("My question")
result = await graph.ainvoke(state)
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=src/vanna_langgraph
```

## Project Structure

```
src/vanna_langgraph/
├── graph/              # LangGraph core
│   ├── state.py       # VannaState schema
│   ├── builder.py     # Graph construction
│   ├── edges.py       # Conditional edge functions
│   └── nodes/         # Node implementations
├── tools/             # LangChain tools
├── providers/         # LLM provider abstraction
├── adapters/          # Compatibility layer
└── utils/             # Helper functions
```

## Documentation

- [Architecture](docs/architecture.md) - Detailed system design
- [Migration Guide](docs/migration.md) - Migrating from original Vanna

## Credits

This project is a refactoring of [Vanna](https://github.com/vanna-ai/vanna) using LangGraph. See [CREDITS.md](CREDITS.md) for acknowledgments.

## License

MIT License - See [LICENSE](LICENSE) for details.
