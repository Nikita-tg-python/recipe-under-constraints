"""POST /recipe: the technologist's free text in, a recipe (or why there is none) out."""

import logging
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.errors import AppError
from app.pipeline import RecipeInfeasible, RecipeOk, UnsupportedProduct, run_recipe
from app.runs import Run

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_REQUEST_CHARS = 2000


class RecipeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)


@router.post("/recipe")
async def create_recipe(
    body: RecipeRequest, request: Request
) -> RecipeOk | RecipeInfeasible | UnsupportedProduct:
    state = request.app.state
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return round((time.perf_counter() - started) * 1000)

    try:
        parsed, outcome = await run_recipe(body.request, state.llm, state.catalog)
    except AppError as exc:
        error = {"error": {"code": exc.code, "message": exc.message}}
        await state.runs.save(Run(body.request, None, None, "error", error, elapsed_ms()))
        logger.warning("recipe run failed: %s (%s ms)", exc.code, elapsed_ms())
        raise

    result = outcome.model_dump(mode="json", exclude={"run_id"})
    run = Run(
        request_text=body.request,
        parsed_constraints=parsed.model_dump(mode="json"),
        matched_template=parsed.matched_template,
        status=outcome.status,
        result=result,
        duration_ms=elapsed_ms(),
    )
    run_id = await state.runs.save(run)
    logger.info(
        "recipe run %s: status=%s template=%s %s ms",
        run_id, outcome.status, parsed.matched_template, run.duration_ms,
    )  # fmt: skip
    return outcome.model_copy(update={"run_id": run_id})
