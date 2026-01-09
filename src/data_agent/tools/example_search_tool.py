"""
Example Search Tool (RAG)

LangChain Tool for searching similar historical question-SQL pairs
to assist in SQL generation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class ExampleSearchInput(BaseModel):
    """Input schema for example search."""
    
    question: str = Field(description="The user's question to find similar examples for")
    top_k: int = Field(default=3, description="Number of similar examples to return")


# RAG store instance (to be configured)
_rag_store: Optional[Any] = None


def set_rag_store(store: Any) -> None:
    """
    Set the RAG store instance for example retrieval.
    
    Args:
        store: RAG store instance (e.g., FAISSStore)
    """
    global _rag_store
    _rag_store = store


def get_rag_store() -> Optional[Any]:
    """Get the current RAG store instance."""
    global _rag_store
    return _rag_store


@tool("search_examples", args_schema=ExampleSearchInput)
def search_examples(question: str, top_k: int = 3) -> str:
    """
    Search for similar historical question-SQL pairs to help generate accurate SQL.
    
    Use this tool BEFORE generating SQL to find relevant examples that can guide
    the SQL generation process. The examples show how similar questions have been
    answered with SQL queries in the past.
    
    Args:
        question: The user's natural language question
        top_k: Number of similar examples to return (default: 3)
        
    Returns:
        A formatted string containing similar question-SQL pairs, or a message
        indicating no similar examples were found.
    """
    global _rag_store
    
    if _rag_store is None:
        return "No RAG store configured. Proceeding without examples."
    
    try:
        # Search for similar examples
        results = _rag_store.search_similar(question, k=top_k)
        
        if not results:
            return "No similar examples found. Generating SQL based on schema only."
        
        # Format results
        formatted_examples = []
        for i, result in enumerate(results, 1):
            example = f"""
Example {i}:
- Question: {result.get('question', 'N/A')}
- SQL: {result.get('sql', 'N/A')}
- Similarity: {result.get('score', 0):.2f}
"""
            formatted_examples.append(example.strip())
        
        header = f"Found {len(results)} similar examples:\n"
        return header + "\n\n".join(formatted_examples)
        
    except Exception as e:
        return f"Error searching examples: {str(e)}. Proceeding without examples."


async def search_examples_async(question: str, top_k: int = 3) -> str:
    """
    Async version of search_examples.
    
    Args:
        question: The user's natural language question
        top_k: Number of similar examples to return
        
    Returns:
        Formatted string with similar examples
    """
    global _rag_store
    
    if _rag_store is None:
        return "No RAG store configured. Proceeding without examples."
    
    try:
        # Check if store has async method
        if hasattr(_rag_store, 'asearch_similar'):
            results = await _rag_store.asearch_similar(question, k=top_k)
        else:
            results = _rag_store.search_similar(question, k=top_k)
        
        if not results:
            return "No similar examples found. Generating SQL based on schema only."
        
        # Format results
        formatted_examples = []
        for i, result in enumerate(results, 1):
            example = f"""
Example {i}:
- Question: {result.get('question', 'N/A')}
- SQL: {result.get('sql', 'N/A')}
- Similarity: {result.get('score', 0):.2f}
"""
            formatted_examples.append(example.strip())
        
        header = f"Found {len(results)} similar examples:\n"
        return header + "\n\n".join(formatted_examples)
        
    except Exception as e:
        return f"Error searching examples: {str(e)}. Proceeding without examples."
