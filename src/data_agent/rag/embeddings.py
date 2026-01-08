"""
Embedding Configuration Module

Provides embedding model configuration for vector stores.
Uses OpenAI embeddings by default with fallback options.
"""

from typing import Optional, List, Callable
from dataclasses import dataclass
import os


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: int = 1536
    
    def __post_init__(self):
        # Auto-detect API key from environment
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


# Global embedding config
_embedding_config: Optional[EmbeddingConfig] = None


def configure_embeddings(config: EmbeddingConfig) -> None:
    """Set the global embedding configuration."""
    global _embedding_config
    _embedding_config = config


def get_embedding_config() -> EmbeddingConfig:
    """Get the current embedding configuration."""
    global _embedding_config
    if _embedding_config is None:
        _embedding_config = EmbeddingConfig()
    return _embedding_config


def get_embedding_function() -> Callable[[str], List[float]]:
    """
    Get an embedding function based on current config.
    
    Returns:
        A function that takes text and returns embedding vector
    """
    config = get_embedding_config()
    
    if config.provider == "openai":
        return _get_openai_embeddings(config)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")


def _get_openai_embeddings(config: EmbeddingConfig) -> Callable[[str], List[float]]:
    """Create OpenAI embedding function."""
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(
        model=config.model,
        openai_api_key=config.api_key,
        openai_api_base=config.base_url,
    )
    
    def embed_text(text: str) -> List[float]:
        return embeddings.embed_query(text)
    
    return embed_text


def get_langchain_embeddings():
    """
    Get LangChain embeddings object for use with FAISS.
    
    Returns:
        LangChain Embeddings object
    """
    config = get_embedding_config()
    
    if config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.model,
            openai_api_key=config.api_key,
            openai_api_base=config.base_url,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
