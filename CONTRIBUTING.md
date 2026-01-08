# Contributing

Guidelines for contributing to Vanna LangGraph.

## Development Setup

```bash
# Clone
git clone <repo>
cd vanna-langgraph

# Install with dev dependencies
pip install -e ".[dev,all-providers]"

# Run tests
pytest
```

## Code Style

- Use **black** for formatting
- Use **ruff** for linting
- Use **mypy** for type checking

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Testing

- Write tests for all new features
- Use pytest fixtures from `conftest.py`
- Mock LLM calls in unit tests

```bash
# Run specific tests
pytest tests/test_graph/test_nodes.py -v

# With coverage
pytest --cov=src/vanna_langgraph --cov-report=html
```

## Adding a New Node

1. Create `src/vanna_langgraph/graph/nodes/my_node.py`
2. Implement async function with `VannaState` signature
3. Add to `nodes/__init__.py`
4. Wire into graph in `builder.py`
5. Add tests in `tests/test_graph/test_nodes.py`

## Adding a New LLM Provider

1. Add provider to `providers/base.py`
2. Add optional dependency to `pyproject.toml`
3. Update documentation

## Pull Requests

1. Create feature branch from `main`
2. Write tests for changes
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with clear description
