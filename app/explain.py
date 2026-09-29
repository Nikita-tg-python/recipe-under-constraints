"""Human explanation of an infeasible request. The LLM only words facts computed by app/relax.py.

The facts (numbers already formatted) go into the prompt; the model is told not to calculate. Its
answer is then checked by code: every number in it must appear in the facts or in the user's own
request, and every recommended number must be mentioned. If the check fails, or the model is
unavailable, the facts themselves are returned as a plain template text. So the explanation never
contains a number that the solver did not produce.
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.llm.base import LLMClient, LLMError, Message
from app.relax import Diagnosis, Relaxation
from app.relax import num as _num
from app.schemas import ParsedRequest

logger = logging.getLogger(__name__)

HEADLINE = "Рецептури, що задовольняє всі обмеження одночасно, немає."
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

SYSTEM_PROMPT = """\
You explain to a food technologist, in Ukrainian, why the requested recipe does not exist and what
could be relaxed. Use only the facts you are given. Never calculate, round, convert or invent
numbers: every number you write must appear in the facts exactly as written there. Mention every
relaxation option, in the given order (the first one is the smallest change); do not recommend
anything that is not in the facts. 2-5 sentences, plain text, no markdown, no greeting.
"""


@dataclass(frozen=True)
class Explanation:
    text: str
    source: Literal["llm", "template"]  # template: model unavailable or its text failed the check


def _relaxation_line(r: Relaxation, position: int) -> str:
    change = f"зміна {_num(r.relative_change * 100, 1)} %"
    cost = f"собівартість такої рецептури {_num(r.recipe.cost_uah_per_kg)} грн/кг"
    if r.constraint == "protein_g_per_100g":
        return (
            f"{position}. Білок: запитано щонайменше {_num(r.requested)} г на 100 г, досяжно "  # type: ignore[arg-type]
            f"щонайменше {_num(r.minimal_feasible)} г на 100 г ({change}); {cost}."  # type: ignore[arg-type]
        )
    if r.constraint == "sugar_reduction_pct":
        return (
            f"{position}. Цукор: запитано на {_num(r.requested)} % менше, ніж у звичайного "  # type: ignore[arg-type]
            f"продукту, досяжно щонайбільше на {_num(r.minimal_feasible)} % менше ({change}); "  # type: ignore[arg-type]
            f"{cost}."
        )
    if r.constraint == "max_ingredients":
        return (
            f"{position}. Кількість інгредієнтів: запитано не більше {r.requested}, потрібно "
            f"щонайменше {r.minimal_feasible} ({change}); {cost}."
        )
    if r.constraint == "cost_ceiling_uah_per_kg":
        return (
            f"{position}. Собівартість: запитано до {_num(r.requested)} грн/кг, найдешевша "  # type: ignore[arg-type]
            f"рецептура з рештою обмежень — від {_num(r.minimal_feasible)} грн/кг ({change})."  # type: ignore[arg-type]
        )
    return (
        f"{position}. Claim «зі зниженим вмістом цукру» (цукру щонайменше на 30 % менше, ніж у "
        f"звичайного продукту): без нього рецептура існує; {cost}."
    )


def fact_lines(diagnosis: Diagnosis) -> list[str]:
    if diagnosis.relaxations:
        return [
            "Варіанти послаблення, кожен окремо (решта обмежень як у запиті), "
            "від найменшої зміни до найбільшої:",
            *(_relaxation_line(r, n) for n, r in enumerate(diagnosis.relaxations, start=1)),
        ]
    return ["Жодне окреме послаблення не допомагає:", *diagnosis.blocking]


def template_text(diagnosis: Diagnosis) -> str:
    return "\n".join([HEADLINE, *fact_lines(diagnosis)])


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in _NUMBER.findall(text)}


def check_numbers(text: str, facts: str, request_text: str, required: list[str]) -> str | None:
    """None if the text is safe to show, else the reason it is not."""
    if not text.strip():
        return "empty text"
    invented = _numbers(text) - _numbers(facts) - _numbers(request_text)
    if invented:
        return f"numbers not in the facts: {sorted(invented)}"
    missing = [n for n in required if float(n.replace(",", ".")) not in _numbers(text)]
    if missing:
        return f"recommended numbers not mentioned: {missing}"
    return None


async def explain_infeasible(
    request: ParsedRequest, diagnosis: Diagnosis, llm: LLMClient
) -> Explanation:
    facts = "\n".join(fact_lines(diagnosis))
    fallback = Explanation(template_text(diagnosis), "template")
    required = [
        _num(r.minimal_feasible)  # type: ignore[arg-type]
        for r in diagnosis.relaxations
        if r.constraint != "sugar_reduced_claim"
    ]
    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", f"Запит технолога: {request.raw_text}\n\nФакти:\n{HEADLINE}\n{facts}"),
    ]
    try:
        text = (await llm.complete(messages)).strip()
    except LLMError as exc:
        logger.warning("explanation: LLM unavailable, template text used: %s", exc)
        return fallback
    problem = check_numbers(text, f"{HEADLINE}\n{facts}", request.raw_text, required)
    if problem:
        logger.warning("explanation: LLM text rejected (%s), template text used", problem)
        return fallback
    return Explanation(text, "llm")
