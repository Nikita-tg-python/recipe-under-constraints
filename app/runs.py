"""recipe_runs: one row per POST /recipe call, whatever the outcome."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import asyncpg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Run:
    request_text: str
    parsed_constraints: dict[str, Any] | None
    matched_template: str | None
    status: str  # ok | infeasible | unsupported_product | error
    result: dict[str, Any]
    duration_ms: int


class RunStore(Protocol):
    async def save(self, run: Run) -> int | None:
        """Row id, or None if the run could not be stored (the answer is still returned)."""
        ...


class PgRunStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, run: Run) -> int | None:
        try:
            async with self._pool.acquire(timeout=5) as conn:
                return await conn.fetchval(
                    """
                    INSERT INTO recipe_runs (request_text, parsed_constraints, matched_template,
                                             status, result, duration_ms)
                    VALUES ($1, $2::jsonb, $3, $4, $5::jsonb, $6)
                    RETURNING id
                    """,
                    run.request_text,
                    None if run.parsed_constraints is None else json.dumps(run.parsed_constraints),
                    run.matched_template,
                    run.status,
                    json.dumps(run.result, ensure_ascii=False),
                    run.duration_ms,
                )
        except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError):
            logger.exception("recipe_runs: could not store the run (status=%s)", run.status)
            return None
