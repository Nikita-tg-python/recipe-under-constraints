"""Free text -> ParsedRequest. The only place where the LLM "understands" the request.

The model returns JSON with facts from the text only: which template, which allergens to exclude,
what protein/cost/claim constraints were stated. It never computes anything (not even unit
conversions): numbers are passed through as written, everything else is left to the solver.

    python -m app.parsing "полуничний йогурт без молока ..."     # live call, prints JSON
"""

import asyncio
import json
import re
import sys
import typing

from pydantic import ValidationError

from app.errors import AppError
from app.llm.base import LLMClient, Message
from app.schemas import Allergen, ParsedRequest, ProductTemplate

MAX_ATTEMPTS = 2  # one retry on invalid JSON / schema violation
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

SYSTEM_PROMPT = """\
You convert a food technologist's request (usually Ukrainian) into JSON constraints for a recipe
solver. You only extract what the text says. Never calculate, estimate or convert numbers — copy
them as stated, and leave everything you are not told as null / false / [].

Supported product templates (pick the closest one by id, or null if none fits):
{templates}

Allergen groups (EU 1169/2011) you may list in allergens_to_exclude:
{allergens}
"без молока" / "безмолочний" / "веганський" / "на рослинній основі" -> "milk" ("безлактозний" is NOT
milk-free: lactose-free products still contain milk, so do not exclude "milk" for it);
"без глютену" -> "gluten"; "без горіхів" -> "nuts" (and "peanuts" only if peanuts are named).

Fields:
- product_type: the product as the user named it, verbatim (e.g. "полуничний йогурт").
- matched_template: template id or null. If null, unmatched_reason: one short sentence in
  Ukrainian why (it is shown to the user).
- allergens_to_exclude: groups the product must not contain.
- protein_constraint:
    "білка не менше, ніж у звичайного" / "як у звичайного"
        -> {{"mode": "at_least_baseline", "value": null}}
    "не менше 8 г білка на 100 г" -> {{"mode": "absolute_g", "value": 8}}
    "підвищений білок" / "більше білка" with no number -> at_least_baseline, and add a note
        that it was read as "not less than the regular product".
    no protein requirement -> null.
- sugar_reduced_claim: true if the label must be able to say "зі зниженим вмістом цукру" /
  "reduced sugar" (or "менше цукру" as a label claim).
- sugar_reduction_pct: a stated amount of sugar reduction versus the regular product, in percent:
  "на 40 % менше цукру" -> 40; "вдвічі менше цукру" -> 50 (the only allowed word-to-number
  reading; any other wording -> null and a note). "зі зниженим вмістом цукру" alone is the label
  claim (sugar_reduced_claim), not a percentage; both may be set. No amount stated -> null.
- max_ingredients: a stated limit on the number of ingredients ("не більше 5 інгредієнтів",
  "максимум 4 компоненти") -> that number; not stated -> null (the service applies its default).
- cost_ceiling_uah_per_kg: the cost limit only if stated per kg in UAH; if stated in another
  unit (per 100 g, per pack) put null and explain in notes — do not convert.
- notes: short remarks, in Ukrainian, about anything in the text you could not map to these fields
  or had to interpret (they are shown to the user), e.g. "«безлактозний» не виключає молоко:
  безлактозних інгредієнтів у каталозі немає".

Answer with one JSON object with exactly these keys:
product_type, matched_template, unmatched_reason, allergens_to_exclude, protein_constraint,
sugar_reduced_claim, sugar_reduction_pct, max_ingredients, cost_ceiling_uah_per_kg, notes.
"""


class ParseError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__("parse_failed", message, 502)


def build_system_prompt(templates: dict[str, ProductTemplate]) -> str:
    listing = "\n".join(f"- {t.id}: {t.name}. {t.description}".rstrip() for t in templates.values())
    allergens = ", ".join(typing.get_args(Allergen))
    return SYSTEM_PROMPT.format(templates=listing, allergens=allergens)


def _validate(raw: str, text: str, templates: dict[str, ProductTemplate]) -> ParsedRequest:
    data = json.loads(_FENCE.sub("", raw))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    data["raw_text"] = text  # set by code, not trusted from the model
    parsed = ParsedRequest.model_validate(data)
    if parsed.matched_template is not None and parsed.matched_template not in templates:
        raise ValueError(
            f"matched_template {parsed.matched_template!r} is not one of {sorted(templates)}"
        )
    return parsed


async def parse_request(
    text: str, llm: LLMClient, templates: dict[str, ProductTemplate]
) -> ParsedRequest:
    if not text.strip():
        raise AppError("empty_request", "Request text is empty", 422)
    messages = [Message("system", build_system_prompt(templates)), Message("user", text)]
    error = ""
    for _ in range(MAX_ATTEMPTS):
        raw = await llm.complete(messages, json_output=True)
        try:
            return _validate(raw, text, templates)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            error = (
                "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            messages = [
                *messages,
                Message("assistant", raw),
                Message("user", f"Your answer was invalid ({error}). Reply with corrected JSON."),
            ]
    raise ParseError(f"could not parse the request after {MAX_ATTEMPTS} attempts: {error}")


def main() -> int:
    from app.config import settings
    from app.llm import create_llm_client
    from app.seed import load_catalog

    text = " ".join(sys.argv[1:]) or sys.stdin.read()
    llm = create_llm_client(settings)
    parsed = asyncio.run(parse_request(text, llm, load_catalog().templates))
    print(f"provider: {llm.provider}", file=sys.stderr)
    print(parsed.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
