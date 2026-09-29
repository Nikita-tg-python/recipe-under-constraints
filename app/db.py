from pathlib import Path

import asyncpg

# Arbitrary constant: serialises migrations when several API workers start at once.
_MIGRATION_LOCK_ID = 4_107_2026


async def create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(database_url, min_size=1, max_size=10)


async def apply_migrations(pool: asyncpg.Pool, migrations_dir: Path) -> list[str]:
    """Run every migrations/*.sql in name order on each start.

    Each file must be idempotent (CREATE ... IF NOT EXISTS etc.), so re-running is a no-op.
    """
    applied = []
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_ID)
        for path in sorted(migrations_dir.glob("*.sql")):
            await conn.execute(path.read_text(encoding="utf-8"))
            applied.append(path.name)
    return applied


async def ping(pool: asyncpg.Pool) -> bool:
    try:
        async with pool.acquire(timeout=2) as conn:
            return await conn.fetchval("SELECT 1") == 1
    except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError):
        return False
