-- Ingredient catalog and product templates. Filled by `python -m app.seed` from data/*.yaml.
-- Applied on every API start: must stay idempotent.

CREATE TABLE IF NOT EXISTS ingredients (
    id                 TEXT PRIMARY KEY,
    name               TEXT    NOT NULL,
    category           TEXT    NOT NULL,
    -- {kcal, protein_g, fat_g, carbs_g, sugar_g, fiber_g, salt_g} per 100 g, EU labelling rules
    nutrients_per_100g JSONB   NOT NULL,
    -- sucrose-equivalent sweetness, grams per 100 g (sugar = 100)
    sweetness_per_100g NUMERIC NOT NULL DEFAULT 0 CHECK (sweetness_per_100g >= 0),
    allergens          TEXT[]  NOT NULL DEFAULT '{}',
    price_per_kg_uah   NUMERIC NOT NULL CHECK (price_per_kg_uah >= 0),
    -- one sentence: where the order of magnitude of the (invented) price comes from
    price_source       TEXT    NOT NULL CHECK (btrim(price_source) <> '')
);

CREATE INDEX IF NOT EXISTS ingredients_category_idx ON ingredients (category);

CREATE TABLE IF NOT EXISTS product_templates (
    id          TEXT PRIMARY KEY,
    name        TEXT  NOT NULL,
    description TEXT  NOT NULL DEFAULT '',
    -- the "regular" product used as the reference for comparative claims: {ingredient_id: grams per 1 kg}
    base_recipe JSONB NOT NULL
);

-- Which ingredient categories a template may use and how many grams of each per 1 kg.
CREATE TABLE IF NOT EXISTS ingredient_categories (
    template_id TEXT    NOT NULL REFERENCES product_templates (id) ON DELETE CASCADE,
    category    TEXT    NOT NULL,
    min_g       NUMERIC NOT NULL,
    max_g       NUMERIC NOT NULL,
    PRIMARY KEY (template_id, category),
    CHECK (min_g >= 0 AND min_g <= max_g AND max_g <= 1000)
);
