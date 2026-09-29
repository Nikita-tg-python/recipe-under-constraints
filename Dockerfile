FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml --extra dev

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
