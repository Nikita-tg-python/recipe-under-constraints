"""Catalog models: ingredients, product templates and the cross-checks between them."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

RECIPE_TOTAL_G = 1000.0
_TOL = 1e-6

# The 14 allergen groups of Reg. (EU) 1169/2011 Annex II.
Allergen = Literal[
    "gluten", "crustaceans", "eggs", "fish", "peanuts", "soy", "milk",
    "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
]  # fmt: skip

_ID = r"^[a-z0-9_]+$"


class Nutrients(BaseModel):
    """Per 100 g, EU labelling rules: carbs exclude fibre, sugars exclude polyols."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kcal: float = Field(ge=0, le=900)
    protein_g: float = Field(ge=0, le=100)
    fat_g: float = Field(ge=0, le=100)
    carbs_g: float = Field(ge=0, le=100)
    sugar_g: float = Field(ge=0, le=100)
    fiber_g: float = Field(ge=0, le=100)
    salt_g: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _consistent(self) -> "Nutrients":
        if self.sugar_g > self.carbs_g + _TOL:
            raise ValueError("sugar_g cannot exceed carbs_g")
        mass = self.protein_g + self.fat_g + self.carbs_g + self.fiber_g + self.salt_g
        if mass > 100 + _TOL:
            raise ValueError(f"nutrients add up to {mass:g} g per 100 g")
        return self


NUTRIENT_KEYS: tuple[str, ...] = tuple(Nutrients.model_fields)


class Ingredient(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    name: str = Field(min_length=1)
    category: str = Field(pattern=_ID)
    nutrients_per_100g: Nutrients
    sweetness_per_100g: float = Field(ge=0)
    allergens: tuple[Allergen, ...] = ()
    price_per_kg_uah: float = Field(ge=0)
    price_source: str = Field(min_length=10)


class CategoryBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_g: float = Field(ge=0, le=RECIPE_TOTAL_G)
    max_g: float = Field(ge=0, le=RECIPE_TOTAL_G)

    @model_validator(mode="after")
    def _ordered(self) -> "CategoryBounds":
        if self.min_g > self.max_g:
            raise ValueError(f"min_g {self.min_g:g} > max_g {self.max_g:g}")
        return self


class ProductTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    name: str = Field(min_length=1)
    description: str = ""
    category_bounds: dict[str, CategoryBounds] = Field(min_length=1)
    # The "regular" product: reference for "not less protein than usual"
    # and for the 30 % sugar threshold.
    base_recipe: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def _self_consistent(self) -> "ProductTemplate":
        if any(g <= 0 for g in self.base_recipe.values()):
            raise ValueError("base_recipe grams must be positive")
        total = sum(self.base_recipe.values())
        if abs(total - RECIPE_TOTAL_G) > _TOL:
            raise ValueError(f"base_recipe adds up to {total:g} g, expected {RECIPE_TOTAL_G:g}")
        lo = sum(b.min_g for b in self.category_bounds.values())
        hi = sum(b.max_g for b in self.category_bounds.values())
        if not lo <= RECIPE_TOTAL_G <= hi:
            raise ValueError(f"category bounds allow {lo:g}–{hi:g} g, cannot reach 1 kg")
        return self


class Catalog(BaseModel):
    """Everything the solver needs, validated as a whole: templates never reference missing data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ingredients: dict[str, Ingredient]
    templates: dict[str, ProductTemplate]

    @model_validator(mode="after")
    def _cross_checks(self) -> "Catalog":
        categories = {i.category for i in self.ingredients.values()}
        for t in self.templates.values():
            if missing := sorted(set(t.category_bounds) - categories):
                raise ValueError(f"template {t.id}: no ingredients in categories {missing}")
            per_category: dict[str, float] = dict.fromkeys(t.category_bounds, 0.0)
            for ing_id, grams in t.base_recipe.items():
                ing = self.ingredients.get(ing_id)
                if ing is None:
                    raise ValueError(f"template {t.id}: unknown ingredient {ing_id!r}")
                if ing.category not in per_category:
                    raise ValueError(
                        f"template {t.id}: {ing_id} is in category {ing.category!r}, "
                        "which the template does not allow"
                    )
                per_category[ing.category] += grams
            for cat, grams in per_category.items():
                b = t.category_bounds[cat]
                if not b.min_g - _TOL <= grams <= b.max_g + _TOL:
                    raise ValueError(
                        f"template {t.id}: base recipe has {grams:g} g of {cat}, "
                        f"outside {b.min_g:g}–{b.max_g:g}"
                    )
        return self


# --- parsed request (KAN-43) ----------------------------------------------------------------


def _as_list(value: Any) -> Any:
    """Models often send null (or a bare string) for an empty list; it means the same."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return value


def _none_as_false(value: Any) -> Any:
    return False if value is None else value


class ProteinConstraint(BaseModel):
    """«білка не менше, ніж у звичайного» -> at_least_baseline; «не менше 8 г білка» -> absolute_g.

    absolute_g is grams of protein per 100 g of product (the unit used on labels).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["at_least_baseline", "absolute_g"]
    value: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def _value_matches_mode(self) -> "ProteinConstraint":
        if self.mode == "absolute_g" and self.value is None:
            raise ValueError("absolute_g needs a value (g of protein per 100 g)")
        if self.mode == "at_least_baseline" and self.value is not None:
            raise ValueError("at_least_baseline takes no value: the baseline is the catalog recipe")
        return self


class ParsedRequest(BaseModel):
    """What the technologist asked for, as structured constraints for the solver.

    Produced by the LLM (facts from the text only, no arithmetic), validated here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    product_type: str = Field(min_length=1)  # as the user named it, e.g. "полуничний йогурт"
    matched_template: str | None  # id of a product template, or None if nothing fits
    unmatched_reason: str | None = None  # why matched_template is None
    allergens_to_exclude: Annotated[list[Allergen], BeforeValidator(_as_list)] = []
    protein_constraint: ProteinConstraint | None = None
    sugar_reduced_claim: Annotated[bool, BeforeValidator(_none_as_false)] = False
    cost_ceiling_uah_per_kg: float | None = Field(default=None, gt=0)
    # «на 40 % менше цукру» / «вдвічі менше цукру»: % less sugar than the regular product
    sugar_reduction_pct: float | None = Field(default=None, gt=0, lt=100)
    # what the model could not map, e.g. a price per 100 g
    notes: Annotated[list[str], BeforeValidator(_as_list)] = []
    raw_text: str

    @model_validator(mode="after")
    def _reason_when_unmatched(self) -> "ParsedRequest":
        if self.matched_template is None and not self.unmatched_reason:
            raise ValueError("unmatched_reason is required when matched_template is null")
        return self
