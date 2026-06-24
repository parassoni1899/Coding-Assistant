"""
ingestion/repo.py — Repository Ingestion
==========================================
Clones a Git repository (shallow) and produces a filtered list of source
files suitable for indexing.

Key design decisions:
  - Shallow clone (--depth 1) keeps disk footprint minimal.
  - File filtering is configurable but defaults to the canonical set for Go
    codebases: .go and .md files, excluding vendor/, testdata/, and *_test.go.
  - Returns a list of RepoFile dataclass instances instead of raw strings so
    that downstream components always have access to both content and metadata.
"""

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import git
from loguru import logger
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepoFile:
    """Represents a single source file extracted from a repository."""

    # Absolute path on disk
    absolute_path: str

    # Path relative to the repository root (used as a stable identifier)
    relative_path: str

    # Raw file content (UTF-8 decoded)
    content: str

    # Detected programming language (go, markdown, python, …)
    language: str

    # Repository root directory
    repo_root: str

    # Byte size of the file
    size_bytes: int = 0

    # Extra metadata slots for downstream enrichment
    extra: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Directories to skip during file walk
SKIP_DIRS: set[str] = {
    "vendor",
    ".git",
    "node_modules",
    "testdata",
    ".cache",
    "dist",
    "build",
    "__pycache__",
}

# File extensions to include and their language labels
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".go": "go",
    ".md": "markdown",
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".java": "java",
}

# Regex patterns for files to always skip (applied to the base filename)
SKIP_FILE_PATTERNS: list[re.Pattern] = [
    re.compile(r".*_test\.go$"),       # Go unit test files
    re.compile(r".*\.pb\.go$"),        # Protobuf generated files
    re.compile(r".*\.gen\.go$"),       # Code-generated files
    re.compile(r"^mock_.*\.go$"),      # Mock files
    re.compile(r".*_mock\.go$"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class RepoIngester:
    """
    Clones a Git repository and iterates over its indexable source files.

    Usage:
        ingester = RepoIngester(
            repo_url="https://github.com/pocketbase/pocketbase",
            local_path="./data/repos/pocketbase",
        )
        files: List[RepoFile] = ingester.ingest()
    """

    def __init__(
        self,
        repo_url: str,
        local_path: str,
        branch: str = "master",
        extensions: Optional[List[str]] = None,
        max_file_size_kb: int = 512,
        force_reclone: bool = False,
    ):
        self.repo_url = repo_url
        self.local_path = Path(local_path).resolve()
        self.branch = branch
        self.extensions = extensions or list(EXTENSION_TO_LANGUAGE.keys())
        self.max_file_bytes = max_file_size_kb * 1024
        self.force_reclone = force_reclone

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self) -> List[RepoFile]:
        """Clone (if needed) and return all indexable RepoFile objects."""
        self._ensure_repo()
        files = list(self._walk_files())
        logger.success(f"Ingestion complete — {len(files)} files found in {self.local_path}")
        return files

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_repo(self) -> None:
        """Clone the repository if not already present, or pull latest changes."""
        if self.force_reclone and self.local_path.exists():
            logger.warning(f"force_reclone=True — removing {self.local_path}")
            shutil.rmtree(self.local_path)

        if not self.local_path.exists():
            logger.info(f"Cloning {self.repo_url} → {self.local_path} (shallow, depth=1)")
            git.Repo.clone_from(
                self.repo_url,
                str(self.local_path),
                depth=1,
                branch=self.branch,
                single_branch=True,
            )
            logger.success("Clone complete.")
        else:
            logger.info(f"Repository already present at {self.local_path} — pulling latest.")
            try:
                repo = git.Repo(str(self.local_path))
                repo.remotes.origin.pull()
            except Exception as exc:
                logger.warning(f"Pull failed (continuing with cached repo): {exc}")

    def _walk_files(self) -> Generator[RepoFile, None, None]:
        """
        Recursively walk the repository directory, yielding RepoFile objects
        for every file that passes the inclusion filters.
        """
        repo_root = str(self.local_path)
        all_paths = []

        for root, dirs, files in os.walk(repo_root):
            # Prune skipped directories in-place to prevent descent
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS and not d.startswith(".")
            ]
            for fname in files:
                all_paths.append(os.path.join(root, fname))

        logger.info(f"Scanning {len(all_paths)} total paths …")

        for abs_path in tqdm(all_paths, desc="Filtering files", unit="file"):
            file = self._try_load_file(abs_path, repo_root)
            if file is not None:
                yield file

    def _try_load_file(self, abs_path: str, repo_root: str) -> Optional[RepoFile]:
        """
        Attempt to load a file. Returns None if the file should be skipped.

        Filters applied (in order):
          1. Extension whitelist
          2. Filename pattern blacklist (test files, generated files)
          3. Maximum file size
          4. UTF-8 decodability
        """
        path = Path(abs_path)
        ext = path.suffix.lower()

        # 1. Extension filter
        if ext not in self.extensions:
            return None

        # 2. Pattern blacklist
        if any(pattern.match(path.name) for pattern in SKIP_FILE_PATTERNS):
            return None

        # 3. Size filter
        size = path.stat().st_size
        if size > self.max_file_bytes:
            logger.debug(f"Skipping large file ({size // 1024}KB): {path.name}")
            return None

        if size == 0:
            return None

        # 4. UTF-8 decoding
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, PermissionError):
            logger.debug(f"Skipping non-UTF-8 / unreadable file: {abs_path}")
            return None

        relative = os.path.relpath(abs_path, repo_root).replace("\\", "/")
        language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

        return RepoFile(
            absolute_path=abs_path,
            relative_path=relative,
            content=content,
            language=language,
            repo_root=repo_root,
            size_bytes=size,
        )
