# Migration Guide

Migrating from original Vanna to Vanna LangGraph.

## Key Differences

| Aspect | Original Vanna | Vanna LangGraph |
|--------|---------------|-----------------|
| Architecture | Agent-based | LangGraph StateGraph |
| Flow Control | Implicit loops | Explicit conditional edges |
| State | ToolContext | Typed VannaState |
| Tools | Custom Tool class | LangChain @tool |
| LLM | LlmService | LangChain ChatModel |

## API Compatibility

Core methods are compatible:

```python
# Original Vanna
vn = VannaBase()
vn.ask("question")
vn.generate_sql("question")
vn.run_sql("SELECT ...")

# Vanna LangGraph (same API)
vn = VannaLangGraph()
vn.ask("question")
vn.generate_sql("question")
vn.run_sql("SELECT ...")
```

## Migration Steps

### 1. Update Imports

```python
# Before
from vanna import VannaBase
from vanna.integrations.openai import OpenAIChat

# After
from vanna_langgraph import VannaLangGraph
```

### 2. Initialize

```python
# Before
class MyVanna(VannaBase, OpenAIChat):
    pass
vn = MyVanna(api_key="...")

# After
vn = VannaLangGraph(
    llm_provider="openai",
    llm_api_key="...",
)
```

### 3. Schema Configuration

```python
# Before
vn.connect_to_sqlite("mydb.db")

# After
from sqlalchemy import create_engine
engine = create_engine("sqlite:///mydb.db")
vn.connect_to_database(engine.connect())
```

## Unavailable Features

Some original Vanna features are not yet implemented:

- **Visualization tools**: Charts and plotting
- **Training data storage**: Persistent examples
- **Web UI**: Flask/Streamlit interfaces

These can be added as extensions to the graph.

## Advanced: Access Graph Directly

For custom workflows, access the underlying graph:

```python
from vanna_langgraph.graph.state import create_initial_state

# Create custom initial state
state = create_initial_state(
    "my question",
    max_retries=5,
    database_schema="...",
)

# Run graph
result = await vn.graph.ainvoke(state)
```

## Getting Help

- Check [Architecture](architecture.md) for design details
- File issues on GitHub
