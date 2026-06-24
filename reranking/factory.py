"""
reranking/factory.py — Reranker Factory
========================================
Provides a clean interface for reranking search results.

Local execution uses `sentence-transformers` with a lightweight
cross-encoder model (runs on CPU efficiently).
Cloud execution maps to Cohere's Rerank API.
"""

import os
from typing import List
from loguru import logger
from config import settings

# Attempt to load the SearchResult type, but avoid circular imports
try:
    from retrieval.vector_store import SearchResult
except ImportError:
    pass


class BaseReranker:
    def rerank(self, query: str, results: List["SearchResult"], top_n: int = 5) -> List["SearchResult"]:
        raise NotImplementedError


class LocalCrossEncoderReranker(BaseReranker):
    """
    Local Reranker using HuggingFace sentence-transformers.
    Calculates a joint attention score over (Query, Document).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info(f"[Reranker] Loading local CrossEncoder: {model_name}")
        # Defaulting to CPU for RAM safety, but handles CUDA if available
        device = "cuda" if _has_cuda() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: List["SearchResult"], top_n: int = 5) -> List["SearchResult"]:
        if not results:
            return []

        # Prepare pairs for the CrossEncoder
        pairs = [[query, r.content] for r in results]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to results and sort
        for result, score in zip(results, scores):
            # Overwrite the original dense search score with the precise rerank score
            result.score = float(score)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]


class CohereCloudReranker(BaseReranker):
    """
    Cloud Reranker using Cohere API.
    """

    def __init__(self):
        import cohere
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is missing for cloud reranker.")
        logger.info("[Reranker] Initializing Cohere Cloud Reranker.")
        self.client = cohere.Client(api_key)

    def rerank(self, query: str, results: List["SearchResult"], top_n: int = 5) -> List["SearchResult"]:
        if not results:
            return []

        docs = [r.content for r in results]
        
        response = self.client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_n,
            return_documents=False,
        )

        # Reorder our local results based on Cohere's index mapping
        reranked_results = []
        for res in response.results:
            original_result = results[res.index]
            original_result.score = res.relevance_score
            reranked_results.append(original_result)

        return reranked_results


def get_reranker() -> BaseReranker:
    """Factory method to get the configured reranker."""
    env = os.getenv("ENVIRONMENT", "local").lower()
    if env == "local":
        return LocalCrossEncoderReranker(model_name=settings.local_rerank_model)
    else:
        provider = os.getenv("CLOUD_RERANK_PROVIDER", "cohere").lower()
        if provider == "cohere":
            return CohereCloudReranker()
        else:
            raise ValueError(f"Unsupported CLOUD_RERANK_PROVIDER: {provider}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
