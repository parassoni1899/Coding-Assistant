"""
agents/tools.py — LangGraph Tools
==================================
Defines the tools available to the Codebase Agent:
- Code Search (Hybrid retrieval)
- File Reader (Read raw file lines)
- Call Graph (Traverse AST relationships)
"""

from pathlib import Path
from typing import List

from langchain_core.tools import tool
from loguru import logger

from config import settings
from graph.builder import GraphBuilder
from retrieval.search import HybridRetriever

# Lazy-loaded instances to avoid circular startup issues
_retriever = None
_graph_db = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_graph_db() -> GraphBuilder:
    global _graph_db
    if _graph_db is None:
        _graph_db = GraphBuilder()
    return _graph_db


@tool
def search_codebase(query: str) -> str:
    """
    Search the codebase using semantic and keyword search.
    Use this to find where a concept or function is defined or discussed.
    
    Args:
        query: A natural language question or symbol name (e.g., 'AuthWithPassword').
    """
    logger.debug(f"[Tool] search_codebase: {query}")
    results = _get_retriever().search(query, top_k=5)
    
    if not results:
        return "No results found for this query."
        
    output = []
    for r in results:
        output.append(
            f"--- [{r.chunk_type}] {r.symbol} (File: {r.file_path}) ---\n{r.content}\n"
        )
    return "\n".join(output)


@tool
def read_file_lines(file_path: str, start_line: int, end_line: int) -> str:
    """
    Reads specific lines from a file in the repository.
    Use this when you know the file path and need to see exact code.
    
    Args:
        file_path: The relative path to the file (e.g., 'core/app.go').
        start_line: The 1-indexed start line.
        end_line: The 1-indexed end line.
    """
    logger.debug(f"[Tool] read_file_lines: {file_path} {start_line}-{end_line}")
    abs_path = Path(settings.repo_local_path) / file_path
    
    if not abs_path.exists():
        return f"Error: File '{file_path}' does not exist in the repository."
        
    try:
        lines = abs_path.read_text(encoding="utf-8").splitlines()
        
        # Adjust to 0-index
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        snippet = lines[start_idx:end_idx]
        formatted = []
        for i, line in enumerate(snippet, start_line):
            formatted.append(f"{i}: {line}")
            
        return "\n".join(formatted)
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def get_call_chain(symbol_name: str, depth: int = 2) -> str:
    """
    Traces the execution path (caller/callee relationships) for a specific function.
    Use this to answer "What happens when X is called?" or "Who calls X?".
    
    Args:
        symbol_name: The exact name of the function/method (e.g., 'CreateRecord').
        depth: How deep to trace the graph (default: 2).
    """
    logger.debug(f"[Tool] get_call_chain: {symbol_name} (depth={depth})")
    try:
        chain = _get_graph_db().get_call_chain(symbol_name, depth)
        if not chain:
            return f"No call chain found for '{symbol_name}'. It might not be indexed or it makes no internal calls."
        
        return f"Call chain for '{symbol_name}':\n" + "\n".join(chain)
    except Exception as exc:
        return f"Error traversing call graph: {exc}"


# List of tools to bind to the LangGraph node
AGENT_TOOLS = [search_codebase, read_file_lines, get_call_chain]
