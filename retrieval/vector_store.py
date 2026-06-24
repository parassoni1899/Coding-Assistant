"""
retrieval/vector_store.py — Vector Store Manager
=================================================
Wraps the configured vector store client (ChromaDB / Qdrant / Pinecone)
and exposes clean ingest + search operations on CodeChunk objects.

The embeddings and store clients are retrieved from config.py factories,
so this module remains environment-agnostic.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm

from chunking.tree_parser import CodeChunk
from config import get_embeddings, get_vector_store, settings


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Number of chunks to send to the embedding API / Ollama per batch.
# Smaller batches keep RAM usage predictable; 32 is safe on 16GB RAM.
EMBED_BATCH_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────────
# Search result model
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult:
    """Wraps a retrieved document with its similarity score."""

    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score

        # Convenience accessors into document.metadata
        self.file_path: str = document.metadata.get("file_path", "")
        self.symbol: str = document.metadata.get("symbol", "")
        self.chunk_type: str = document.metadata.get("chunk_type", "")
        self.package: str = document.metadata.get("package", "")
        self.start_line: int = int(document.metadata.get("start_line", 0))
        self.end_line: int = int(document.metadata.get("end_line", 0))
        self.content: str = document.page_content

    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"symbol={self.symbol!r}, "
            f"file={self.file_path!r})"
        )

    def to_context_block(self) -> str:
        """
        Format this result as a context block for injection into the LLM prompt.
        Includes source attribution metadata so the LLM can cite correctly.
        """
        lines = [
            f"--- [{self.chunk_type.upper()}] {self.symbol} ---",
            f"File: {self.file_path}  (lines {self.start_line}–{self.end_line})",
            f"Package: {self.package}",
            "",
            self.content,
            "",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Manager
# ─────────────────────────────────────────────────────────────────────────────

class VectorStoreManager:
    """
    High-level interface for ingesting and querying code chunks.

    Lifecycle:
        manager = VectorStoreManager()
        manager.ingest_chunks(chunks)      # Build the index
        results = manager.search(query)    # Search
    """

    def __init__(self):
        self._embeddings = get_embeddings()
        self._store = get_vector_store(self._embeddings)
        logger.info("VectorStoreManager initialized.")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_chunks(self, chunks: List[CodeChunk], clear_existing: bool = False) -> int:
        """
        Embed and store a list of CodeChunks.

        Args:
            chunks:         List of CodeChunk objects from the chunking pipeline.
            clear_existing: If True, drop all existing documents before ingesting.
                            Set to True only on a full re-index.

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            logger.warning("ingest_chunks called with empty list.")
            return 0

        if clear_existing:
            logger.warning("Clearing existing vector store data …")
            self._clear_store()

        # Convert CodeChunks to LangChain Documents
        documents: List[Document] = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.to_document_text(),
                metadata=chunk.to_metadata(),
            )
            documents.append(doc)

        # Batch ingest to avoid overwhelming the embedding model
        total_stored = 0
        batches = _batch(documents, EMBED_BATCH_SIZE)

        logger.info(f"Ingesting {len(documents)} chunks in batches of {EMBED_BATCH_SIZE} …")
        for batch in tqdm(batches, desc="Embedding & storing", unit="batch"):
            try:
                ids = self._store.add_documents(batch)
                total_stored += len(ids)
            except Exception as exc:
                logger.error(f"Batch ingest failed: {exc}")
                # Continue with next batch rather than aborting
                time.sleep(1)

        logger.success(f"Stored {total_stored}/{len(documents)} chunks.")
        return total_stored

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Dense vector similarity search.

        Args:
            query:           Natural language or code query string.
            top_k:           Number of results to return (defaults to RETRIEVAL_TOP_K).
            filter_metadata: Optional dict of metadata filters, e.g.
                             {"chunk_type": "func", "package": "apis"}.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        top_k = top_k or settings.retrieval_top_k

        try:
            if filter_metadata:
                docs_and_scores = self._store.similarity_search_with_score(
                    query, k=top_k, filter=filter_metadata
                )
            else:
                docs_and_scores = self._store.similarity_search_with_score(
                    query, k=top_k
                )
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

        results = [SearchResult(doc, score) for doc, score in docs_and_scores]
        logger.debug(f"Dense search returned {len(results)} results for: {query[:60]!r}")
        return results

    def search_by_metadata(
        self,
        filter_metadata: Dict[str, Any],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Retrieve chunks purely by metadata filter (no vector search).
        Useful for exact lookups like "give me all chunks in package 'apis'".
        """
        try:
            docs = self._store.get(where=filter_metadata, limit=top_k)  # ChromaDB API
        except AttributeError:
            # Fallback for stores that don't support .get()
            docs = self._store.similarity_search(
                "", k=top_k, filter=filter_metadata
            )
        if isinstance(docs, dict):
            # ChromaDB .get() returns dict with 'documents' / 'metadatas' keys
            results = []
            for content, meta in zip(docs.get("documents", []), docs.get("metadatas", [])):
                doc = Document(page_content=content, metadata=meta)
                results.append(SearchResult(doc, score=1.0))
            return results
        return [SearchResult(d, score=1.0) for d in docs]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _clear_store(self) -> None:
        """Attempt to clear all documents from the configured store."""
        try:
            # ChromaDB exposes a reset method via the underlying client
            if hasattr(self._store, "_collection"):
                self._store._collection.delete(where={"chunk_id": {"$ne": ""}})
                logger.info("ChromaDB collection cleared.")
            else:
                logger.warning(
                    "Clear not implemented for this store backend. "
                    "Delete the persist directory manually and re-ingest."
                )
        except Exception as exc:
            logger.error(f"Failed to clear store: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _batch(items: List[Any], size: int):
    """Split a list into fixed-size batches."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
