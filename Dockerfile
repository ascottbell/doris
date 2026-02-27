# Pin to specific digest for reproducible builds. To update:
#   docker pull python:3.12-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
ARG BASE_IMAGE=python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab

# =============================================================================
# Stage 1: Build — install compilers and build native extensions
# =============================================================================
FROM ${BASE_IMAGE} AS builder

WORKDIR /build

# System deps for compiling native extensions (cryptography, sqlite-vec, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a virtual env so we can copy the whole tree
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker.txt requirements-docker.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-docker.lock

# =============================================================================
# Stage 2: Runtime — no compilers, no build tools
# =============================================================================
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy pre-built virtualenv from builder (no build-essential/git in this layer)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Data directory — matches the docker-compose volume mount at /app/data
ENV DORIS_DATA_DIR=/app/data

# Create non-root user and data/log directories with correct ownership
RUN useradd --create-home --shell /bin/bash doris && \
    mkdir -p /app/data /app/logs && \
    chown -R doris:doris /app/data /app/logs

# Copy application code (owned by doris, not root)
COPY --chown=doris:doris . .

# Run as non-root user
USER doris

# Default port
EXPOSE 8000

# Health check against the unauthenticated /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "main.py"]
