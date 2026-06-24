"""
graph/builder.py — AST Graph Database
======================================
Builds a local SQLite database that maps caller/callee relationships
and package dependencies based on the AST parser output.

This enables our LangGraph agent to answer questions like:
"What happens when CreateRecord is called?" by traversing the graph.
"""

import os
import re
import sqlite3
from typing import List

from loguru import logger

from chunking.tree_parser import CodeChunk
from config import settings


class GraphBuilder:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.graph_db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                name TEXT,
                type TEXT,
                file_path TEXT,
                package TEXT,
                start_line INTEGER,
                end_line INTEGER
            );

            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_id INTEGER,
                callee_name TEXT,
                FOREIGN KEY(caller_id) REFERENCES symbols(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_symbol_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_callee_name ON calls(callee_name);
        """)
        self.conn.commit()

    def build_from_chunks(self, chunks: List[CodeChunk]):
        """
        Populate the graph database using extracted CodeChunks.
        In a production implementation, the AST Go parser would emit exact
        call-sites. Here, we use a regex fallback over the raw chunk content
        to identify calls to other symbols for the MVP.
        """
        logger.info(f"Building SQLite Call Graph from {len(chunks)} chunks...")
        cursor = self.conn.cursor()

        # Clear existing data
        cursor.executescript("DELETE FROM calls; DELETE FROM symbols;")

        # Insert symbols
        for chunk in chunks:
            # We only track functions and methods as active nodes
            if chunk.chunk_type not in ("func", "method"):
                continue

            cursor.execute("""
                INSERT INTO symbols (chunk_id, name, type, file_path, package, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id, chunk.symbol, chunk.chunk_type,
                chunk.file_path, chunk.package, chunk.start_line, chunk.end_line
            ))
            
            caller_id = cursor.lastrowid
            
            # Simple heuristic to extract callees: Look for words followed by '('
            # This is an MVP approximation. A true implementation parses the AST call expressions.
            callees = set(re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\s*\(', chunk.content))
            
            for callee in callees:
                if callee == chunk.symbol: # Ignore self-recursion
                    continue
                cursor.execute("INSERT INTO calls (caller_id, callee_name) VALUES (?, ?)", (caller_id, callee))

        self.conn.commit()
        
        # Log stats
        cursor.execute("SELECT COUNT(*) FROM symbols")
        num_symbols = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM calls")
        num_calls = cursor.fetchone()[0]
        
        logger.success(f"Graph Database built: {num_symbols} symbols, {num_calls} call edges.")

    def get_callees(self, symbol_name: str) -> List[str]:
        """Find functions called BY the given symbol."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT c.callee_name 
            FROM calls c
            JOIN symbols s ON c.caller_id = s.id
            WHERE s.name = ?
        """, (symbol_name,))
        return [row[0] for row in cursor.fetchall()]

    def get_callers(self, callee_name: str) -> List[str]:
        """Find functions that CALL the given symbol."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT s.name, s.file_path, s.start_line
            FROM calls c
            JOIN symbols s ON c.caller_id = s.id
            WHERE c.callee_name = ?
        """, (callee_name,))
        
        # Return format: "FunctionName (file.go:123)"
        return [f"{row[0]} ({row[1]}:{row[2]})" for row in cursor.fetchall()]

    def get_call_chain(self, symbol_name: str, depth: int = 2) -> List[str]:
        """
        Uses a recursive CTE to trace a dependency chain up to N levels.
        """
        query = """
            WITH RECURSIVE dependency_chain(id, name, path, depth) AS (
                SELECT id, name, file_path, 0 
                FROM symbols 
                WHERE name = ?
                
                UNION ALL
                
                SELECT s.id, s.name, s.file_path, dc.depth + 1
                FROM symbols s
                JOIN calls c ON s.name = c.callee_name
                JOIN dependency_chain dc ON c.caller_id = dc.id
                WHERE dc.depth < ?
            )
            SELECT DISTINCT name, path, depth FROM dependency_chain ORDER BY depth;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (symbol_name, depth))
        
        chain = []
        for row in cursor.fetchall():
            indent = "  " * row[2]
            chain.append(f"{indent}→ {row[0]} ({row[1]})")
        return chain
