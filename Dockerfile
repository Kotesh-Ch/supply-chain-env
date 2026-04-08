# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements-lite.txt
requirements.txt
# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY requirements.txt .

# Install dependencies into a virtual env
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
       --no-cache -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY environment.py  .
COPY server.py       .
COPY graders.py      .
COPY inference.py    .
COPY openenv.yaml    .

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 hfuser && chown -R hfuser /app
USER hfuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start server
CMD ["python", "server.py"]
