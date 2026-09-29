"""A tiny artificial catalog with simple numbers: expected values in tests are computed by hand."""

from app.schemas import Catalog, ParsedRequest


def ingredient(id_, category, price, sweetness=0, allergens=(), **nutrients):
    values = dict(kcal=0, protein_g=0, fat_g=0, carbs_g=0, sugar_g=0, fiber_g=0, salt_g=0)
    return {
        "id": id_, "name": id_, "category": category, "nutrients_per_100g": values | nutrients,
        "sweetness_per_100g": sweetness, "allergens": list(allergens), "price_per_kg_uah": price,
        "price_source": "test catalog, simple numbers",
    }  # fmt: skip


TINY = Catalog.model_validate(
    {
        "ingredients": {
            i["id"]: i
            for i in [
                ingredient(
                    "milk",
                    "base",
                    30,
                    allergens=["milk"],
                    kcal=50,
                    protein_g=3,
                    carbs_g=5,
                    sugar_g=5,
                ),
                ingredient("oat_drink", "base", 50, kcal=45, protein_g=1, carbs_g=7, sugar_g=4),
                ingredient(
                    "sugar", "sweetener", 20, sweetness=100, kcal=400, carbs_g=100, sugar_g=100
                ),
                ingredient("stevia", "sweetener", 1000, sweetness=10000),
                ingredient(
                    "cream",
                    "dairy",
                    90,
                    allergens=["milk"],
                    kcal=200,
                    protein_g=2.5,
                    fat_g=20,
                    carbs_g=3.5,
                    sugar_g=3.5,
                ),
            ]  # fmt: skip
        },
        "templates": {
            "drink": {
                "id": "drink",
                "name": "Солодкий напій",
                "category_bounds": {
                    "base": {"min_g": 800, "max_g": 1000},
                    "sweetener": {"min_g": 0, "max_g": 200},
                },
                # regular: protein 2.7, sugar 14.5, kcal 85, sweetness 10 per 100 g; 29 UAH/kg
                "base_recipe": {"milk": 900, "sugar": 100},
            },
            # a template whose "dairy" category exists only with milk: excluding milk empties it
            "creamy": {
                "id": "creamy",
                "name": "Вершковий напій",
                "category_bounds": {
                    "dairy": {"min_g": 100, "max_g": 200},
                    "base": {"min_g": 700, "max_g": 900},
                    "sweetener": {"min_g": 0, "max_g": 100},
                },
                "base_recipe": {"cream": 150, "milk": 800, "sugar": 50},
            },
        },
    }
)
DRINK = TINY.templates["drink"]


def request(template="drink", **kw) -> ParsedRequest:
    return ParsedRequest.model_validate(
        {"product_type": template, "matched_template": template, "raw_text": "test", **kw}
    )
