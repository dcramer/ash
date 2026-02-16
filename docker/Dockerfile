# Ash - Personal Assistant Agent
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.6.3 /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Create data directory and non-root user
RUN useradd -m -s /bin/bash -u 1000 ash && \
    mkdir -p /app/data && \
    chown -R ash:ash /app/data

# Set environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run as non-root user
USER ash

# Default command
CMD ["uv", "run", "ash", "serve", "--host", "0.0.0.0", "--port", "8080"]
