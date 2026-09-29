# Stage 1: dependencies only, pinned by uv.lock. This layer is rebuilt only when pyproject.toml or
# uv.lock change, not on every code edit. Dev tools (pytest, ruff) are included: `make test`,
# `make lint` and `make proof` run inside this image, so the host needs nothing but Docker.
FROM python:3.12-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra dev --no-cache

# Stage 2: runtime = the ready virtualenv + the code; no uv, no build cache.
FROM python:3.12-slim

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY --from=deps /opt/venv /opt/venv

WORKDIR /app
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
