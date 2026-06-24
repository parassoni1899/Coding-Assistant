"""
chunking/parser.py — Chunking Orchestrator
===========================================
Drives the full chunking pipeline:

  1. Dispatches files to the appropriate language parser.
  2. For Go files: invokes the native `ast_extractor` binary via subprocess
     and converts its JSON output to CodeChunk objects.
  3. For other files: falls back to a Markdown / plain-text splitter.
  4. Returns a flat list of CodeChunk objects ready for embedding.

Why a native Go binary instead of tree-sitter?
  - The standard library `go/ast` package is the canonical, battle-tested Go
    parser. It handles generics, build tags, and all edge cases correctly.
  - Tree-sitter Go bindings in Python are good but occasionally lag behind
    language spec changes. Using the official parser avoids this entirely.
  - The binary is small (~2MB), compiles in seconds, and runs fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

from chunking.tree_parser import BaseParser, CodeChunk
from ingestion.repo import RepoFile


# ─────────────────────────────────────────────────────────────────────────────
# Go AST Parser (calls native binary)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the compiled binary (relative to project root)
_GO_BINARY_NAME = "ast_extractor.exe" if sys.platform == "win32" else "ast_extractor"
_GO_BINARY_PATH = Path(__file__).parent.parent / "graph" / _GO_BINARY_NAME
_GO_SOURCE_PATH = Path(__file__).parent.parent / "graph" / "ast_extractor.go"


def _ensure_go_binary() -> Path:
    """
    Compile the Go AST extractor binary if it doesn't exist or if the source
    file is newer than the binary.
    Requires `go` to be installed and on PATH.
    """
    source_mtime = _GO_SOURCE_PATH.stat().st_mtime if _GO_SOURCE_PATH.exists() else 0
    binary_mtime = _GO_BINARY_PATH.stat().st_mtime if _GO_BINARY_PATH.exists() else 0

    if binary_mtime >= source_mtime and _GO_BINARY_PATH.exists():
        return _GO_BINARY_PATH

    logger.info(f"Compiling Go AST extractor: {_GO_BINARY_PATH}")
    result = subprocess.run(
        ["go", "build", "-o", str(_GO_BINARY_PATH), str(_GO_SOURCE_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile ast_extractor.go:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    logger.success(f"Go binary compiled: {_GO_BINARY_PATH}")
    return _GO_BINARY_PATH


def _run_go_extractor(repo_root: str) -> List[dict]:
    """
    Run the compiled Go binary against the repository root.
    Returns the parsed JSON list of symbol dicts.
    """
    binary = _ensure_go_binary()
    result = subprocess.run(
        [str(binary), repo_root],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        logger.error(f"ast_extractor stderr:\n{result.stderr}")
        raise RuntimeError("ast_extractor binary failed.")

    if result.stderr:
        # Binary prints WARNs to stderr but still succeeds
        for line in result.stderr.strip().splitlines():
            logger.debug(f"[ast_extractor] {line}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse ast_extractor JSON output: {exc}")


class GoASTParser(BaseParser):
    """
    Parses Go source files using the native ast_extractor binary.
    Converts JSON symbol dicts to CodeChunk objects.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".go"]

    def parse_repository(self, repo_root: str) -> List[CodeChunk]:
        """
        Parse the ENTIRE repository at once (more efficient than file-by-file
        because the binary walks the tree itself).
        """
        logger.info(f"Running Go AST extractor on {repo_root} …")
        raw_symbols = _run_go_extractor(repo_root)
        logger.info(f"Extracted {len(raw_symbols)} raw symbols.")

        chunks: List[CodeChunk] = []
        for sym in raw_symbols:
            content = sym.get("content", "").strip()
            if not content:
                continue

            # Make file_path relative to repo root
            abs_path = sym.get("file_path", "")
            try:
                rel_path = os.path.relpath(abs_path, repo_root).replace("\\", "/")
            except ValueError:
                rel_path = abs_path

            chunk = CodeChunk(
                file_path=rel_path,
                package=sym.get("package", ""),
                symbol=sym.get("name", ""),
                chunk_type=sym.get("type", "func"),
                language="go",
                content=content,
                start_line=sym.get("start_line", 0),
                end_line=sym.get("end_line", 0),
                receiver=sym.get("receiver") or None,
                imports=sym.get("imports") or [],
            )
            chunks.append(chunk)

        return chunks

    def parse_file(self, file_path: str, content: str, repo_root: str) -> List[CodeChunk]:
        """
        Single-file parsing is not efficient with the Go binary.
        This method is provided for interface compliance — use parse_repository
        for bulk processing.
        """
        # Run against the file's parent directory (small scope)
        file_dir = str(Path(file_path).parent)
        return self.parse_repository(file_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown / Plain-text Fallback Parser
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownParser(BaseParser):
    """
    Naive recursive character splitter for Markdown documentation files.
    Produces 'file'-type CodeChunk objects with sensible size limits.
    """

    # Markdown files tend to be documentation — keep chunks larger
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )

    @property
    def supported_extensions(self) -> List[str]:
        return [".md"]

    def parse_file(self, file_path: str, content: str, repo_root: str) -> List[CodeChunk]:
        splits = self._splitter.split_text(content)
        chunks = []
        for i, text in enumerate(splits):
            if not text.strip():
                continue
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    package="",
                    symbol=f"{Path(file_path).stem}_chunk_{i}",
                    chunk_type="file",
                    language="markdown",
                    content=text.strip(),
                    start_line=0,
                    end_line=0,
                )
            )
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main Chunking Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingPipeline:
    """
    Orchestrates the chunking stage for a list of ingested RepoFile objects.

    For Go files, uses GoASTParser.parse_repository() for efficiency.
    For other files, delegates to the appropriate single-file parser.
    """

    def __init__(self, min_content_length: int = 30):
        self._go_parser = GoASTParser()
        self._md_parser = MarkdownParser()
        self._min_length = min_content_length

        # Map extension → parser for non-Go files
        self._file_parsers: Dict[str, BaseParser] = {
            ".md": self._md_parser,
        }

    def chunk_repository(self, files: List[RepoFile]) -> List[CodeChunk]:
        """
        Process all ingested files and return a unified list of CodeChunk objects.

        Strategy:
          - Go files: batch-process using the native Go binary (faster).
          - Other files: process one-by-one using Python parsers.
        """
        if not files:
            return []

        # Split files by language
        go_files = [f for f in files if f.language == "go"]
        other_files = [f for f in files if f.language != "go"]

        all_chunks: List[CodeChunk] = []

        # ── Go: batch parse via native binary ────────────────────────────────
        if go_files:
            repo_root = go_files[0].repo_root
            try:
                go_chunks = self._go_parser.parse_repository(repo_root)
                logger.info(f"Go parser produced {len(go_chunks)} chunks.")
                all_chunks.extend(go_chunks)
            except RuntimeError as exc:
                logger.error(f"Go AST parsing failed: {exc}")
                logger.warning("Falling back to full-file chunks for Go files.")
                all_chunks.extend(self._fallback_file_chunks(go_files))

        # ── Other languages: file-by-file ─────────────────────────────────
        if other_files:
            for repo_file in tqdm(other_files, desc="Parsing non-Go files", unit="file"):
                ext = Path(repo_file.relative_path).suffix.lower()
                parser = self._file_parsers.get(ext)
                if parser is None:
                    continue
                try:
                    chunks = parser.parse_file(
                        file_path=repo_file.relative_path,
                        content=repo_file.content,
                        repo_root=repo_file.repo_root,
                    )
                    all_chunks.extend(chunks)
                except Exception as exc:
                    logger.warning(f"Parser error for {repo_file.relative_path}: {exc}")

        # ── Post-process: filter empty / too-short chunks ─────────────────
        before = len(all_chunks)
        all_chunks = [
            c for c in all_chunks
            if c.content and len(c.content.strip()) >= self._min_length
        ]
        removed = before - len(all_chunks)
        if removed:
            logger.debug(f"Removed {removed} chunks below minimum length ({self._min_length} chars).")

        logger.success(f"Chunking complete — {len(all_chunks)} chunks ready for embedding.")
        return all_chunks

    def _fallback_file_chunks(self, files: List[RepoFile]) -> List[CodeChunk]:
        """Produce one CodeChunk per file as a fallback when AST parsing fails."""
        chunks = []
        for f in files:
            if not f.content.strip():
                continue
            chunks.append(
                CodeChunk(
                    file_path=f.relative_path,
                    package="",
                    symbol=Path(f.relative_path).stem,
                    chunk_type="file",
                    language=f.language,
                    content=f.content[:4000],  # Truncate very large files
                    start_line=1,
                    end_line=f.content.count("\n") + 1,
                )
            )
        return chunks

    @staticmethod
    def compute_stats(chunks: List[CodeChunk]) -> dict:
        """Return summary statistics useful for debugging and evaluation."""
        if not chunks:
            return {}
        type_counts: Dict[str, int] = {}
        for c in chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
        lengths = [len(c.content) for c in chunks]
        return {
            "total": len(chunks),
            "by_type": type_counts,
            "avg_length": round(sum(lengths) / len(lengths)),
            "max_length": max(lengths),
            "min_length": min(lengths),
        }
