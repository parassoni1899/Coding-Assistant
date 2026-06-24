# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for Codebase Assistant
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build the Go AST Parser binary
FROM golang:1.22-alpine AS go-builder
WORKDIR /build
# Copy only the Go source
COPY graph/ast_extractor.go .
# Compile native static binary
RUN CGO_ENABLED=0 GOOS=linux go build -o ast_extractor ast_extractor.go


# Stage 2: Python Runtime Environment
FROM python:3.11-slim

# Install system dependencies (git for ingestion)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-built Go binary from stage 1 into the python graph/ folder
COPY --from=go-builder /build/ast_extractor ./graph/ast_extractor

# Copy the rest of the application
COPY . .

# Ensure the DB paths exist
RUN mkdir -p /app/data/repos /app/data/chroma_db

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default command starts the API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
