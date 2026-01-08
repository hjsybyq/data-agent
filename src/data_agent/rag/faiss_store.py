"""
FAISS Vector Store Implementation

Provides FAISS-based vector storage for question-SQL examples.
Uses LangChain's FAISS wrapper for convenience.
"""

from typing import Optional, List, Dict, Any
import os
import json
from pathlib import Path

from data_agent.rag.base import Example, Documentation, VectorStore
from data_agent.rag.embeddings import get_langchain_embeddings


class FAISSStore:
    """
    FAISS-based vector store for RAG.
    
    Stores question-SQL examples and documentation for retrieval.
    Supports persistence to disk.
    
    Example:
        store = FAISSStore(persist_dir="./vanna_rag")
        store.add_example("瀹㈡埛鏁伴噺", "SELECT COUNT(*) FROM Customer")
        examples = store.search_examples("鏈夊灏戝鎴凤紵", k=3)
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        Initialize FAISS store.
        
        Args:
            persist_dir: Directory for persistence (None for in-memory only)
            auto_save: Whether to auto-save after modifications
        """
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.auto_save = auto_save
        
        # Metadata storage (FAISS doesn't store metadata natively)
        self._examples_metadata: Dict[str, Example] = {}
        self._docs_metadata: Dict[str, Documentation] = {}
        
        # FAISS indices (lazy loaded)
        self._examples_index = None
        self._docs_index = None
        
        # Load from disk if exists
        if self.persist_dir and self.persist_dir.exists():
            self._load()
    
    def _get_embeddings(self):
        """Get the embeddings model."""
        return get_langchain_embeddings()
    
    def _ensure_examples_index(self):
        """Ensure examples index is initialized."""
        if self._examples_index is None:
            from langchain_community.vectorstores import FAISS
            # Create empty index with a dummy document
            embeddings = self._get_embeddings()
            self._examples_index = FAISS.from_texts(
                ["__init__"],
                embeddings,
                metadatas=[{"id": "__init__", "type": "init"}]
            )
    
    def _ensure_docs_index(self):
        """Ensure documentation index is initialized."""
        if self._docs_index is None:
            from langchain_community.vectorstores import FAISS
            embeddings = self._get_embeddings()
            self._docs_index = FAISS.from_texts(
                ["__init__"],
                embeddings,
                metadatas=[{"id": "__init__", "type": "init"}]
            )
    
    def add_example(
        self,
        question: str,
        sql: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a question-SQL example."""
        example = Example.create(question, sql, metadata)
        
        self._ensure_examples_index()
        
        # Add to FAISS index
        self._examples_index.add_texts(
            [question],
            metadatas=[{"id": example.id, "type": "example"}]
        )
        
        # Store full metadata
        self._examples_metadata[example.id] = example
        
        if self.auto_save and self.persist_dir:
            self._save()
        
        return example.id
    
    def search_examples(
        self,
        query: str,
        k: int = 3,
    ) -> List[Example]:
        """Search for similar examples."""
        if self._examples_index is None or len(self._examples_metadata) == 0:
            return []
        
        # Search FAISS
        results = self._examples_index.similarity_search_with_score(query, k=k + 1)
        
        examples = []
        for doc, score in results:
            example_id = doc.metadata.get("id")
            if example_id and example_id != "__init__" and example_id in self._examples_metadata:
                example = self._examples_metadata[example_id]
                # FAISS returns L2 distance, convert to similarity
                # Lower distance = higher similarity
                example.score = 1.0 / (1.0 + score)
                examples.append(example)
        
        return examples[:k]
    
    def add_documentation(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add documentation."""
        import uuid
        doc_id = str(uuid.uuid4())
        doc = Documentation(
            id=doc_id,
            content=content,
            metadata=metadata or {},
        )
        
        self._ensure_docs_index()
        
        self._docs_index.add_texts(
            [content],
            metadatas=[{"id": doc_id, "type": "documentation"}]
        )
        
        self._docs_metadata[doc_id] = doc
        
        if self.auto_save and self.persist_dir:
            self._save()
        
        return doc_id
    
    def search_documentation(
        self,
        query: str,
        k: int = 3,
    ) -> List[Documentation]:
        """Search for relevant documentation."""
        if self._docs_index is None or len(self._docs_metadata) == 0:
            return []
        
        results = self._docs_index.similarity_search_with_score(query, k=k + 1)
        
        docs = []
        for doc, score in results:
            doc_id = doc.metadata.get("id")
            if doc_id and doc_id != "__init__" and doc_id in self._docs_metadata:
                documentation = self._docs_metadata[doc_id]
                documentation.score = 1.0 / (1.0 + score)
                docs.append(documentation)
        
        return docs[:k]
    
    def get_all_examples(self) -> List[Example]:
        """Get all stored examples."""
        return list(self._examples_metadata.values())
    
    def remove_example(self, example_id: str) -> bool:
        """Remove an example by ID."""
        if example_id in self._examples_metadata:
            del self._examples_metadata[example_id]
            # Note: FAISS doesn't support efficient deletion
            # Would need to rebuild index for true deletion
            if self.auto_save and self.persist_dir:
                self._save()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._examples_metadata.clear()
        self._docs_metadata.clear()
        self._examples_index = None
        self._docs_index = None
        
        if self.persist_dir:
            import shutil
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
    
    def _save(self) -> None:
        """Save to disk."""
        if not self.persist_dir:
            return
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        if self._examples_index:
            self._examples_index.save_local(str(self.persist_dir / "examples_index"))
        if self._docs_index:
            self._docs_index.save_local(str(self.persist_dir / "docs_index"))
        
        # Save metadata
        metadata = {
            "examples": {
                k: {
                    "id": v.id,
                    "question": v.question,
                    "sql": v.sql,
                    "metadata": v.metadata,
                }
                for k, v in self._examples_metadata.items()
            },
            "documentation": {
                k: {
                    "id": v.id,
                    "content": v.content,
                    "metadata": v.metadata,
                }
                for k, v in self._docs_metadata.items()
            }
        }
        
        with open(self.persist_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _load(self) -> None:
        """Load from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return
        
        # Load metadata
        metadata_path = self.persist_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            for k, v in metadata.get("examples", {}).items():
                self._examples_metadata[k] = Example(
                    id=v["id"],
                    question=v["question"],
                    sql=v["sql"],
                    metadata=v.get("metadata", {}),
                )
            
            for k, v in metadata.get("documentation", {}).items():
                self._docs_metadata[k] = Documentation(
                    id=v["id"],
                    content=v["content"],
                    metadata=v.get("metadata", {}),
                )
        
        # Load FAISS indices
        examples_path = self.persist_dir / "examples_index"
        docs_path = self.persist_dir / "docs_index"
        
        if examples_path.exists():
            from langchain_community.vectorstores import FAISS
            self._examples_index = FAISS.load_local(
                str(examples_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        
        if docs_path.exists():
            from langchain_community.vectorstores import FAISS
            self._docs_index = FAISS.load_local(
                str(docs_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
    
    @property
    def example_count(self) -> int:
        """Number of stored examples."""
        return len(self._examples_metadata)
    
    @property
    def documentation_count(self) -> int:
        """Number of stored documentation entries."""
        return len(self._docs_metadata)
