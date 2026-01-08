"""
RAG (Retrieval Augmented Generation) Module

Provides vector store infrastructure for storing and retrieving
question-SQL examples to enhance SQL generation accuracy.
"""

from data_agent.rag.base import (
    Example,
    Documentation,
    VectorStore,
)
from data_agent.rag.faiss_store import FAISSStore
from data_agent.rag.embeddings import (
    EmbeddingConfig,
    configure_embeddings,
    get_embedding_config,
)

__all__ = [
    "Example",
    "Documentation",
    "VectorStore",
    "FAISSStore",
    "EmbeddingConfig",
    "configure_embeddings",
    "get_embedding_config",
]
