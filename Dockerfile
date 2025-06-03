# EasyMLOps Production Dockerfile
# Multi-stage build for optimized container size and security

# Build stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Create and set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r easymlops && useradd -r -g easymlops -m easymlops

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories with proper permissions
RUN mkdir -p /app/models /app/bentos /app/logs /app/static \
    && chown -R easymlops:easymlops /app

# Copy application code
COPY --chown=easymlops:easymlops app/ ./app/
COPY --chown=easymlops:easymlops static/ ./static/
COPY --chown=easymlops:easymlops alembic/ ./alembic/
COPY --chown=easymlops:easymlops alembic.ini ./

# Copy startup scripts
COPY --chown=easymlops:easymlops scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# Switch to non-root user
USER easymlops

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 