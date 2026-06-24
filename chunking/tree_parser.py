"""
chunking/tree_parser.py — Abstract AST Parser Interface
========================================================
Defines the CodeChunk dataclass and the abstract BaseParser class.

All language-specific parsers (Go, Python, TypeScript …) implement
BaseParser. This keeps the orchestration layer (parser.py) decoupled
from language specifics, making it trivial to add new languages.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core data model — the atom flowing through the pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    """
    A single, semantically-complete code unit extracted from a source file.

    This is the fundamental atom that flows through the entire pipeline:
      ingestion → chunking → embedding → vector store → retrieval → LLM

    Fields map directly to the metadata stored alongside each vector.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    # Repository-relative file path (used as a stable, human-readable reference)
    file_path: str

    # Go / Python package or module name
    package: str

    # Symbol name: function name, struct name, interface name, etc.
    symbol: str

    # One of: "func" | "method" | "struct" | "interface" | "file"
    chunk_type: str

    # Programming language
    language: str

    # ── Content ───────────────────────────────────────────────────────────────

    # Raw source code of this chunk (the text that gets embedded)
    content: str

    # ── Source location ───────────────────────────────────────────────────────

    start_line: int = 0
    end_line: int = 0

    # ── Relationships ─────────────────────────────────────────────────────────

    # For methods: the receiver type (e.g., "*PocketBase", "App")
    receiver: Optional[str] = None

    # Import paths visible in the file where this chunk lives
    imports: List[str] = field(default_factory=list)

    # ── Computed / enriched later ─────────────────────────────────────────────

    # Unique identifier built during storage (file_path:symbol:start_line)
    chunk_id: str = ""

    # Any extra metadata added by downstream enrichment steps
    extra: dict = field(default_factory=dict)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = f"{self.file_path}:{self.symbol}:{self.start_line}"

    def to_metadata(self) -> dict:
        """
        Serialize this chunk's metadata fields for vector store storage.

        The 'content' field is excluded because it is stored as the document
        body; only the searchable / filterable metadata goes here.
        """
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "package": self.package,
            "symbol": self.symbol,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "receiver": self.receiver or "",
            # Store imports as a comma-separated string for ChromaDB compatibility
            # (ChromaDB metadata values must be str | int | float | bool)
            "imports": ",".join(self.imports),
        }

    def to_document_text(self) -> str:
        """
        Build the text representation that gets embedded.

        We prepend a header line with file path and symbol name so that
        the embedding model encodes location context alongside code.
        This dramatically improves retrieval precision for exact-name queries.
        """
        header = f"// File: {self.file_path} | Symbol: {self.symbol} | Type: {self.chunk_type}\n"
        return header + self.content


# ─────────────────────────────────────────────────────────────────────────────
# Abstract parser interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseParser(abc.ABC):
    """
    Abstract base class for language-specific AST parsers.

    Concrete implementations must override `parse_file`.
    """

    @abc.abstractmethod
    def parse_file(self, file_path: str, content: str, repo_root: str) -> List[CodeChunk]:
        """
        Parse a single source file and return a list of CodeChunk objects.

        Args:
            file_path: Repository-relative path to the file.
            content:   Raw UTF-8 content of the file.
            repo_root: Absolute path to the repository root (for computing
                       relative paths from absolute ones returned by parsers).

        Returns:
            A list of CodeChunk objects. May be empty if the file contains
            no extractable symbols (e.g., a blank file or a config file).
        """
        ...

    @property
    @abc.abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return the file extensions this parser handles (e.g., ['.go'])."""
        ...
