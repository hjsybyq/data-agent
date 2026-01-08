"""
Example Retrieval Node

Retrieves similar question-SQL examples from the vector store
for few-shot prompting in SQL generation.
"""

from typing import Dict, Any, Optional, List

from data_agent.graph.state import AgentState


# Global vector store reference (set by adapter)
_vector_store = None


def set_vector_store(store) -> None:
    """Set the global vector store reference."""
    global _vector_store
    _vector_store = store


def get_vector_store():
    """Get the global vector store."""
    return _vector_store


async def example_retrieval(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve similar question-SQL examples for few-shot prompting.
    
    This node searches the vector store for examples similar to the
    current question and adds them to the state for use by SQL generation.
    
    Args:
        state: Current graph state with normalized_question
        
    Returns:
        State update with retrieved_examples
    """
    global _vector_store
    
    # Get the question to search with
    question = state.get("normalized_question") or state.get("user_question", "")
    
    if not question:
        return {"retrieved_examples": []}
    
    # If no vector store configured, return empty
    if _vector_store is None:
        return {"retrieved_examples": []}
    
    # Search for similar examples
    try:
        examples = _vector_store.search_examples(question, k=3)
        
        # Format for use in prompts
        retrieved = [
            {
                "question": ex.question,
                "sql": ex.sql,
                "score": ex.score,
            }
            for ex in examples
            if ex.score > 0.1  # Filter low-confidence matches
        ]
        
        return {"retrieved_examples": retrieved}
    
    except Exception as e:
        # Log error but don't fail the pipeline
        import logging
        logging.warning(f"Example retrieval failed: {e}")
        return {"retrieved_examples": []}


def format_examples_for_prompt(examples: List[Dict[str, Any]]) -> str:
    """
    Format retrieved examples for inclusion in SQL generation prompt.
    
    Args:
        examples: List of retrieved examples
        
    Returns:
        Formatted string for prompt
    """
    if not examples:
        return ""
    
    parts = ["## Similar Examples\n"]
    parts.append("Here are some similar questions and their SQL queries:\n\n")
    
    for i, ex in enumerate(examples, 1):
        parts.append(f"**Example {i}:**\n")
        parts.append(f"Question: {ex['question']}\n")
        parts.append(f"SQL:\n```sql\n{ex['sql']}\n```\n\n")
    
    return "".join(parts)
