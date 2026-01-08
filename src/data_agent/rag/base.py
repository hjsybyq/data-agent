"""
RAG Base Module

Defines the protocol and models for vector store implementations.
"""

from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Example:
    """
    Represents a question-SQL example for retrieval.
    
    Attributes:
        id: Unique identifier for the example
        question: The natural language question
        sql: The corresponding SQL query
        metadata: Additional metadata (source, tags, etc.)
        score: Similarity score from retrieval (0-1, higher is more similar)
    """
    id: str
    question: str
    sql: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    
    @classmethod
    def create(
        cls,
        question: str,
        sql: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Example":
        """Create a new example with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            question=question,
            sql=sql,
            metadata=metadata or {},
        )


@dataclass
class Documentation:
    """
    Represents documentation for retrieval.
    
    Attributes:
        id: Unique identifier
        content: The documentation text
        metadata: Additional metadata
        score: Similarity score from retrieval
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol for vector store implementations.
    
    Implementations must provide methods for adding and searching
    examples and documentation.
    """
    
    def add_example(
        self,
        question: str,
        sql: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a question-SQL example to the store.
        
        Args:
            question: The natural language question
            sql: The corresponding SQL query
            metadata: Optional metadata
            
        Returns:
            The ID of the added example
        """
        ...
    
    def search_examples(
        self,
        query: str,
        k: int = 3,
    ) -> List[Example]:
        """
        Search for similar examples.
        
        Args:
            query: The query to search for
            k: Number of results to return
            
        Returns:
            List of similar examples with scores
        """
        ...
    
    def add_documentation(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add documentation to the store.
        
        Args:
            content: The documentation text
            metadata: Optional metadata
            
        Returns:
            The ID of the added documentation
        """
        ...
    
    def search_documentation(
        self,
        query: str,
        k: int = 3,
    ) -> List[Documentation]:
        """
        Search for relevant documentation.
        
        Args:
            query: The query to search for
            k: Number of results to return
            
        Returns:
            List of relevant documentation with scores
        """
        ...
    
    def get_all_examples(self) -> List[Example]:
        """Get all stored examples."""
        ...
    
    def remove_example(self, example_id: str) -> bool:
        """
        Remove an example by ID.
        
        Returns:
            True if removed, False if not found
        """
        ...
    
    def clear(self) -> None:
        """Clear all stored data."""
        ...
