"""
config.py — Provider Factory Abstractions
==========================================
Reads .env variables and returns the correct concrete client implementation
for LLM, Embeddings, VectorStore, and Reranker.

Switching between local and cloud is done purely via .env:
    ENVIRONMENT=local   → Ollama + ChromaDB + Local Cross-Encoder
    ENVIRONMENT=cloud   → OpenAI/Claude + Pinecone + Cohere Rerank
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    """Read an environment variable with a default fallback."""
    return os.getenv(key, default)


def _require_env(key: str) -> str:
    """Read a required environment variable; raise if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Please configure it in your .env file."
        )
    return value


ENVIRONMENT = _env("ENVIRONMENT", "local").lower()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm():
    """
    Returns the configured LangChain Chat model.

    Local:  ChatOllama (qwen2.5-coder:7b or similar)
    Cloud:  ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI
    """
    if ENVIRONMENT == "local":
        from langchain_ollama import ChatOllama
        model = _env("LOCAL_LLM_MODEL", "qwen2.5-coder:7b")
        base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"[LLM] Local Ollama → model={model}")
        return ChatOllama(model=model, base_url=base_url, temperature=0)

    else:  # cloud
        provider = _env("CLOUD_LLM_PROVIDER", "openai").lower()
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            model = _env("OPENAI_LLM_MODEL", "gpt-4o-mini")
            api_key = _require_env("OPENAI_API_KEY")
            logger.info(f"[LLM] Cloud OpenAI → model={model}")
            return ChatOpenAI(model=model, api_key=api_key, temperature=0)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            model = _env("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022")
            api_key = _require_env("ANTHROPIC_API_KEY")
            logger.info(f"[LLM] Cloud Anthropic → model={model}")
            return ChatAnthropic(model=model, api_key=api_key, temperature=0)

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = _env("GEMINI_LLM_MODEL", "gemini-1.5-flash")
            api_key = _require_env("GEMINI_API_KEY")
            logger.info(f"[LLM] Cloud Gemini → model={model}")
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)

        else:
            raise ValueError(f"Unsupported CLOUD_LLM_PROVIDER: '{provider}'. "
                             f"Choose from: openai, anthropic, gemini")


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings Factory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns the configured LangChain Embeddings implementation.

    Local:
        ollama      → OllamaEmbeddings (nomic-embed-text, 8k context window)
        huggingface → HuggingFaceEmbeddings (BAAI/bge-large-en-v1.5, 512 ctx)
    Cloud:
        openai      → OpenAIEmbeddings (text-embedding-3-small)
        cohere      → CohereEmbeddings (embed-english-v3.0)
    """
    if ENVIRONMENT == "local":
        provider = _env("LOCAL_EMBEDDING_PROVIDER", "ollama").lower()

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            model = _env("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
            base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info(f"[Embeddings] Local Ollama → model={model}")
            return OllamaEmbeddings(model=model, base_url=base_url)

        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            model = _env("LOCAL_HF_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
            logger.info(f"[Embeddings] Local HuggingFace → model={model}")
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            raise ValueError(f"Unsupported LOCAL_EMBEDDING_PROVIDER: '{provider}'")

    else:  # cloud
        provider = _env("CLOUD_EMBEDDING_PROVIDER", "openai").lower()

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = _require_env("OPENAI_API_KEY")
            logger.info("[Embeddings] Cloud OpenAI → text-embedding-3-small")
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

        elif provider == "cohere":
            from langchain_cohere import CohereEmbeddings
            api_key = _require_env("COHERE_API_KEY")
            logger.info("[Embeddings] Cloud Cohere → embed-english-v3.0")
            return CohereEmbeddings(
                model="embed-english-v3.0",
                cohere_api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported CLOUD_EMBEDDING_PROVIDER: '{provider}'")


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store(embeddings=None):
    """
    Returns a configured LangChain VectorStore.

    Accepts an optional embeddings instance; falls back to get_embeddings().
    This is NOT cached because the collection name can change per call.

    Local:  ChromaDB (persisted SQLite-backed store)
    Cloud:  Qdrant Cloud | Pinecone Serverless
    """
    if embeddings is None:
        embeddings = get_embeddings()

    backend = _env("VECTOR_STORE", "chroma").lower()

    if backend == "chroma":
        from langchain_community.vectorstores import Chroma
        persist_dir = _env("CHROMA_PERSIST_DIR", "./data/chroma_db")
        collection = _env("CHROMA_COLLECTION_NAME", "codebase_assistant")
        logger.info(f"[VectorStore] ChromaDB → collection={collection}, path={persist_dir}")
        return Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    elif backend == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        url = _env("QDRANT_URL", "http://localhost:6333")
        collection = _env("QDRANT_COLLECTION_NAME", "codebase_assistant")
        client = QdrantClient(url=url)
        logger.info(f"[VectorStore] Qdrant → collection={collection}, url={url}")
        return QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )

    elif backend == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        api_key = _require_env("PINECONE_API_KEY")
        index_name = _require_env("PINECONE_INDEX_NAME")
        logger.info(f"[VectorStore] Pinecone → index={index_name}")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=api_key,
        )

    else:
        raise ValueError(f"Unsupported VECTOR_STORE: '{backend}'. "
                         f"Choose from: chroma, qdrant, pinecone")


# ─────────────────────────────────────────────────────────────────────────────
# Settings snapshot (convenience dataclass for passing config around)
# ─────────────────────────────────────────────────────────────────────────────

class Settings:
    """Centralised, pre-parsed settings accessible throughout the codebase."""
    environment: str = ENVIRONMENT

    # Ingestion
    repo_url: str = _env("TARGET_REPO_URL", "https://github.com/pocketbase/pocketbase")
    repo_local_path: str = _env("TARGET_REPO_LOCAL_PATH", "./data/repos/pocketbase")

    # Retrieval
    retrieval_top_k: int = int(_env("RETRIEVAL_TOP_K", "20"))
    rerank_top_m: int = int(_env("RERANK_TOP_M", "5"))
    bm25_index_path: str = _env("BM25_INDEX_PATH", "./data/bm25_index.pkl")

    # Graph DB
    graph_db_path: str = _env("GRAPH_DB_PATH", "./data/graph.db")

    # Reranker
    local_rerank_model: str = _env("LOCAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


settings = Settings()
