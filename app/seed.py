"""Load data/*.yaml into Postgres: `python -m app.seed`.

Idempotent: ingredients and templates are upserted by id, rows no longer in the YAML are deleted,
so after every run the tables mirror the files exactly.
"""

import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

import asyncpg
import yaml
from pydantic import ValidationError

from app import db
from app.config import PROJECT_ROOT, settings
from app.schemas import Catalog

DATA_DIR = PROJECT_ROOT / "data"


def load_catalog(data_dir: Path = DATA_DIR) -> Catalog:
    ingredients = yaml.safe_load((data_dir / "ingredients.yaml").read_text(encoding="utf-8"))
    templates = yaml.safe_load((data_dir / "templates.yaml").read_text(encoding="utf-8"))
    for kind, rows in (("ingredient", ingredients), ("template", templates)):
        ids = [r["id"] for r in rows]
        if dupes := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate {kind} ids: {dupes}")
    return Catalog.model_validate(
        {
            "ingredients": {r["id"]: r for r in ingredients},
            "templates": {r["id"]: r for r in templates},
        }
    )


def _num(x: float) -> Decimal:
    return Decimal(str(x))


async def write_catalog(conn: asyncpg.Connection, catalog: Catalog) -> None:
    async with conn.transaction():
        await conn.executemany(
            """
            INSERT INTO ingredients (id, name, category, nutrients_per_100g, sweetness_per_100g,
                                     allergens, price_per_kg_uah, price_source)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                nutrients_per_100g = EXCLUDED.nutrients_per_100g,
                sweetness_per_100g = EXCLUDED.sweetness_per_100g,
                allergens = EXCLUDED.allergens,
                price_per_kg_uah = EXCLUDED.price_per_kg_uah,
                price_source = EXCLUDED.price_source
            """,
            [
                (
                    i.id,
                    i.name,
                    i.category,
                    i.nutrients_per_100g.model_dump_json(),
                    _num(i.sweetness_per_100g),
                    list(i.allergens),
                    _num(i.price_per_kg_uah),
                    i.price_source,
                )
                for i in catalog.ingredients.values()
            ],
        )
        await conn.execute(
            "DELETE FROM ingredients WHERE NOT (id = ANY($1::text[]))", list(catalog.ingredients)
        )

        await conn.executemany(
            """
            INSERT INTO product_templates (id, name, description, base_recipe)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                base_recipe = EXCLUDED.base_recipe
            """,
            [
                (t.id, t.name, t.description, json.dumps(t.base_recipe))
                for t in catalog.templates.values()
            ],
        )
        # Cascades to ingredient_categories of removed templates.
        await conn.execute(
            "DELETE FROM product_templates WHERE NOT (id = ANY($1::text[]))",
            list(catalog.templates),
        )

        # Bounds are a pure function of the template: replace them wholesale.
        await conn.execute("DELETE FROM ingredient_categories")
        await conn.executemany(
            """
            INSERT INTO ingredient_categories (template_id, category, min_g, max_g)
            VALUES ($1, $2, $3, $4)
            """,
            [
                (t.id, cat, _num(b.min_g), _num(b.max_g))
                for t in catalog.templates.values()
                for cat, b in t.category_bounds.items()
            ],
        )


async def main() -> int:
    try:
        catalog = load_catalog()
    except (ValidationError, ValueError, KeyError) as exc:
        print(f"invalid catalog in {DATA_DIR}:\n{exc}", file=sys.stderr)
        return 1

    pool = await db.create_pool(settings.database_url)
    try:
        await db.apply_migrations(pool, settings.migrations_dir)
        async with pool.acquire() as conn:
            await write_catalog(conn, catalog)
            counts = {
                table: await conn.fetchval(f"SELECT count(*) FROM {table}")
                for table in ("ingredients", "product_templates", "ingredient_categories")
            }
    finally:
        await pool.close()

    print(", ".join(f"{table}: {n}" for table, n in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
