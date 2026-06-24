"""
tests/test_chunking.py — Unit tests for the Chunking Pipeline
==============================================================
Run with: pytest tests/test_chunking.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking.parser import ChunkingPipeline, MarkdownParser
from ingestion.repo import RepoFile


def test_markdown_parser():
    parser = MarkdownParser()
    content = "# Title\n\nSome text.\n\n## Subtitle\n\nMore text."
    
    chunks = parser.parse_file(
        file_path="docs/readme.md",
        content=content,
        repo_root="/fake/root"
    )
    
    assert len(chunks) > 0
    assert chunks[0].language == "markdown"
    assert chunks[0].chunk_type == "file"
    assert "Title" in chunks[0].content


def test_chunking_pipeline_skips_empty():
    pipeline = ChunkingPipeline(min_content_length=10)
    
    empty_file = RepoFile(
        absolute_path="/fake/root/empty.md",
        relative_path="empty.md",
        content="   ", # Empty after strip
        language="markdown",
        repo_root="/fake/root"
    )
    
    short_file = RepoFile(
        absolute_path="/fake/root/short.md",
        relative_path="short.md",
        content="short", # Length 5 < min_content_length 10
        language="markdown",
        repo_root="/fake/root"
    )
    
    valid_file = RepoFile(
        absolute_path="/fake/root/valid.md",
        relative_path="valid.md",
        content="This is a valid long file.",
        language="markdown",
        repo_root="/fake/root"
    )
    
    chunks = pipeline.chunk_repository([empty_file, short_file, valid_file])
    
    assert len(chunks) == 1
    assert "valid long file" in chunks[0].content
