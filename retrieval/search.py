"""
retrieval/search.py — Hybrid Search Engine
===========================================
Combines Dense Vector Search (ChromaDB) with Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF).

The output is then re-ordered by the cross-encoder Reranker to provide
the highest quality context chunks for the LLM.
"""

import os
import pickle
from typing import Dict, List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from config import settings
from retrieval.vector_store import SearchResult, VectorStoreManager
from reranking.factory import get_reranker


class HybridRetriever:
    """
    Orchestrates Vector Search, BM25 Search, RRF Fusion, and Reranking.
    """

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.reranker = get_reranker()
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_results_cache: List[SearchResult] = []
        
        self._load_bm25_index()

    def _load_bm25_index(self):
        """Loads the BM25 index from disk if available."""
        if os.path.exists(settings.bm25_index_path):
            try:
                with open(settings.bm25_index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_index = data["index"]
                    self.bm25_results_cache = data["documents"]
                logger.info("Loaded BM25 sparse index.")
            except Exception as exc:
                logger.warning(f"Failed to load BM25 index: {exc}")

    def build_bm25_index(self, results: List[SearchResult]):
        """
        Builds and saves the BM25 index over all stored chunks.
        Called during ingestion.
        """
        if not results:
            return

        logger.info(f"Building BM25 index over {len(results)} chunks...")
        # Tokenize by splitting on whitespace (a real implementation might use a code tokenizer)
        tokenized_corpus = [res.content.split() for res in results]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_results_cache = results

        # Persist to disk
        os.makedirs(os.path.dirname(settings.bm25_index_path), exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump({
                "index": self.bm25_index,
                "documents": self.bm25_results_cache
            }, f)
        logger.success("BM25 index built and saved.")

    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Full Retrieval Pipeline:
        1. Dense Search (top_k)
        2. Sparse Search (BM25, top_k)
        3. RRF Fusion
        4. Cross-Encoder Reranking (top_m)
        """
        top_k = top_k or settings.retrieval_top_k
        top_m = settings.rerank_top_m

        # 1. Dense Search
        logger.debug("Executing Dense Vector Search...")
        dense_results = self.vector_manager.search(query, top_k=top_k, filter_metadata=filter_metadata)

        # 2. Sparse Search (BM25)
        sparse_results = []
        if self.bm25_index and not filter_metadata:
            # BM25 is skipped if metadata filters are applied (for simplicity in MVP)
            logger.debug("Executing Sparse BM25 Search...")
            tokenized_query = query.split()
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top K indices
            top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            for idx in top_n_idx:
                if scores[idx] > 0:  # Only include if there's a keyword match
                    res = self.bm25_results_cache[idx]
                    res.score = scores[idx] # Temporary BM25 score
                    sparse_results.append(res)

        # 3. Reciprocal Rank Fusion (RRF)
        fused_results = self._rrf(dense_results, sparse_results)
        
        # Limit fused results to top_k before reranking to save time
        fused_results = fused_results[:top_k]

        # 4. Reranking
        logger.debug(f"Reranking top {len(fused_results)} candidates...")
        final_results = self.reranker.rerank(query, fused_results, top_n=top_m)

        return final_results

    def _rrf(self, dense: List[SearchResult], sparse: List[SearchResult], k=60) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion.
        Maps chunk_id (or content hash if no chunk_id) to RRF score.
        """
        scores = {}
        doc_map = {}

        def add_ranks(results_list: List[SearchResult]):
            for rank, res in enumerate(results_list):
                # Use chunk_id if available, otherwise file_path + start_line
                doc_id = res.document.metadata.get("chunk_id", f"{res.file_path}:{res.start_line}")
                doc_map[doc_id] = res
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        add_ranks(dense)
        add_ranks(sparse)

        # Sort by RRF score descending
        sorted_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        fused = []
        for doc_id, rrf_score in sorted_ids:
            res = doc_map[doc_id]
            res.score = rrf_score # Update to RRF score temporarily
            fused.append(res)

        return fused
