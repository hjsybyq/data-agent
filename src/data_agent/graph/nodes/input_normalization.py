"""
Input Normalization Node

This node preprocesses the user's natural language question,
standardizing terminology and extracting key entities.
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from data_agent.graph.state import AgentState


async def input_normalization(state: AgentState) -> Dict[str, Any]:
    """
    Normalize and preprocess the user's question.
    
    This node:
    - Cleans up the input text
    - Expands abbreviations if known
    - Standardizes common terms
    - Identifies potential table/column references
    
    Args:
        state: Current graph state with user_question
        
    Returns:
        State updates with normalized_question
    """
    user_question = state["user_question"]
    
    # Basic normalization
    normalized = user_question.strip()
    
    # Remove multiple spaces
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Common SQL-related term expansions
    expansions = {
        r'\bnum\b': 'number',
        r'\bavg\b': 'average',
        r'\bmax\b': 'maximum',
        r'\bmin\b': 'minimum',
        r'\bqty\b': 'quantity',
        r'\bamt\b': 'amount',
        r'\byoy\b': 'year over year',
        r'\bmom\b': 'month over month',
        r'\bytd\b': 'year to date',
        r'\bmtd\b': 'month to date',
    }
    
    for pattern, replacement in expansions.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    # Ensure it ends with proper punctuation for clarity
    if normalized and not normalized[-1] in '.?!':
        normalized = normalized + '?'
    
    return {
        "normalized_question": normalized,
    }
