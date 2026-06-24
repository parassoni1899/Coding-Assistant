"""
main.py — CLI Entry Point
==========================
Provides a clean command-line interface for all pipeline stages.

Commands:
    ingest      Clone the repository and build the vector index.
    search      Run a one-off search query and display results.
    chat        Interactive Q&A session in the terminal.
    stats       Show index statistics.

Usage:
    python main.py ingest
    python main.py search "Where is authentication implemented?"
    python main.py chat
    python main.py stats

Requires .env to be configured. Copy .env.example → .env first.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

# Add project root to sys.path (allows running from any directory)
sys.path.insert(0, str(Path(__file__).parent))

from config import settings

app = typer.Typer(
    name="codebase-assistant",
    help="🤖 Codebase Assistant RAG — Semantic search and Q&A over a Git repository.",
    add_completion=False,
)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    repo_url: str = typer.Option(
        None,
        "--repo", "-r",
        help="Git repository URL (overrides TARGET_REPO_URL in .env)",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force re-clone even if the repository already exists locally.",
    ),
    clear: bool = typer.Option(
        False,
        "--clear", "-c",
        help="Clear the existing vector index before ingesting.",
    ),
):
    """
    Clone the target repository and build the vector search index.

    This is the first command you must run before searching or chatting.
    It performs: clone → parse (AST) → embed → store.
    """
    from ingestion.repo import RepoIngester
    from chunking.parser import ChunkingPipeline
    from retrieval.vector_store import VectorStoreManager

    url = repo_url or settings.repo_url
    local_path = settings.repo_local_path

    console.print(Panel.fit(
        f"[bold cyan]📥 Ingesting repository[/bold cyan]\n"
        f"URL:        {url}\n"
        f"Local path: {local_path}\n"
        f"Force clone: {force}",
        title="Codebase Assistant — Ingest",
    ))

    # Step 1: Clone / pull
    ingester = RepoIngester(
        repo_url=url,
        local_path=local_path,
        force_reclone=force,
    )
    with console.status("[cyan]Cloning repository …[/cyan]"):
        files = ingester.ingest()
    console.print(f"[green]✓[/green] {len(files)} files loaded from repository.")

    # Step 2: Chunk (AST-aware)
    pipeline = ChunkingPipeline()
    with console.status("[cyan]Parsing and chunking source files …[/cyan]"):
        chunks = pipeline.chunk_repository(files)

    stats = ChunkingPipeline.compute_stats(chunks)
    _print_chunk_stats(stats)

    # Step 3: Embed and store
    manager = VectorStoreManager()
    with console.status("[cyan]Embedding and storing chunks …[/cyan]"):
        stored = manager.ingest_chunks(chunks, clear_existing=clear)

    console.print(f"[green]✓[/green] {stored} chunks stored in vector index.")
    console.print("[bold green]✅ Ingestion complete! Run `python main.py chat` to start.[/bold green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (natural language or code symbol name)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to show"),
    pkg: Optional[str] = typer.Option(
        None, "--package", "-p",
        help="Filter results to a specific Go package name",
    ),
    chunk_type: Optional[str] = typer.Option(
        None, "--type", "-t",
        help="Filter by chunk type: func | method | struct | interface | file",
    ),
):
    """
    Run a single semantic search query and print matching code chunks.
    """
    from retrieval.search import HybridRetriever

    console.print(f"\n[bold]🔍 Searching:[/bold] {query}\n")

    manager = HybridRetriever()

    # Build optional metadata filter
    metadata_filter = {}
    if pkg:
        metadata_filter["package"] = pkg
    if chunk_type:
        metadata_filter["chunk_type"] = chunk_type

    results = manager.search(
        query=query,
        top_k=top_k,
        filter_metadata=metadata_filter or None,
    )

    if not results:
        console.print("[yellow]No results found. Make sure you have run `python main.py ingest` first.[/yellow]")
        return

    for i, result in enumerate(results, 1):
        console.print(
            Panel(
                Syntax(result.content, "go", theme="monokai", line_numbers=True),
                title=(
                    f"[bold]{i}.[/bold] "
                    f"[cyan]{result.symbol}[/cyan] "
                    f"[dim]({result.chunk_type})[/dim] · "
                    f"[yellow]{result.file_path}[/yellow] "
                    f"[dim]L{result.start_line}–{result.end_line}[/dim] · "
                    f"score={result.score:.4f}"
                ),
                border_style="blue",
            )
        )


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show retrieved context before the answer"),
):
    """
    Start an interactive Q&A session. Type 'exit' or 'quit' to stop.
    """
    from retrieval.search import HybridRetriever
    from config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    console.print(Panel.fit(
        "[bold cyan]🤖 Codebase Assistant — Interactive Chat[/bold cyan]\n"
        "Ask questions about the indexed codebase in natural language.\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
        title="Chat Session",
    ))

    manager = HybridRetriever()
    llm = get_llm()

    SYSTEM_PROMPT = """You are an expert Principal Software Engineer with deep knowledge of Go codebases.
Your task is to answer questions about the provided codebase based ONLY on the retrieved code snippets below.

CRITICAL RULES:
1. Answer ONLY from the provided context. Do NOT hallucinate code or logic that is not shown.
2. If the context does not contain enough information, say: "I cannot find the relevant implementation in the current context."
3. Always cite your sources: mention the file path and function/struct name for every claim you make.
4. When explaining request flows, list steps chronologically, referencing file paths and function names.
5. Be concise but technically precise. Assume the user is an experienced backend engineer."""

    while True:
        try:
            query = console.input("\n[bold green]You >[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting chat.[/dim]")
            break

        if not query or query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        # Retrieve context
        with console.status("[cyan]Searching codebase …[/cyan]"):
            results = manager.search(query, top_k=settings.rerank_top_m)

        if not results:
            console.print("[yellow]No relevant context found.[/yellow]")
            continue

        # Build context string
        context_blocks = "\n".join(r.to_context_block() for r in results)

        if verbose:
            console.print("\n[dim]─── Retrieved Context ───[/dim]")
            console.print(context_blocks)
            console.print("[dim]─────────────────────────[/dim]\n")

        # Generate answer
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"<context>\n{context_blocks}\n</context>\n\n<question>{query}</question>"),
        ]

        with console.status("[cyan]Generating answer …[/cyan]"):
            response = llm.invoke(messages)

        console.print(f"\n[bold blue]Assistant >[/bold blue] {response.content}\n")

        # Print source references
        console.print("[dim]Sources:[/dim]")
        for r in results:
            console.print(f"  [dim]• {r.file_path}:{r.start_line} — {r.symbol}[/dim]")


@app.command()
def stats():
    """Show current vector index statistics."""
    from retrieval.vector_store import VectorStoreManager
    manager = VectorStoreManager()

    # Query for a count by doing a broad search (ChromaDB doesn't expose .count() on the LangChain wrapper)
    try:
        store = manager._store
        if hasattr(store, "_collection"):
            count = store._collection.count()
            console.print(f"[green]Total chunks in index:[/green] {count}")
        else:
            console.print("[yellow]Stats not available for this backend.[/yellow]")
    except Exception as exc:
        console.print(f"[red]Error fetching stats: {exc}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_chunk_stats(stats: dict) -> None:
    if not stats:
        return
    table = Table(title="Chunking Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total chunks", str(stats.get("total", 0)))
    table.add_row("Avg length (chars)", str(stats.get("avg_length", 0)))
    table.add_row("Max length (chars)", str(stats.get("max_length", 0)))
    table.add_row("Min length (chars)", str(stats.get("min_length", 0)))
    for chunk_type, count in stats.get("by_type", {}).items():
        table.add_row(f"  → {chunk_type}", str(count))
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
