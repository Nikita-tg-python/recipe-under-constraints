"""Proof harness: do the recipes the service returns really satisfy the request?

For every request in eval/requests.jsonl the script calls POST /recipe (a running server with --url,
or the app in-process with the real LLM from .env) and checks the answer on its own:
- it never trusts `satisfied` and imports nothing from the solver: nutrients, cost and allergens are
  recomputed here from the raw data/*.yaml, with this file's own arithmetic;
- besides the service's constraints_check, the recipe is checked against the expectations written
  by hand in requests.jsonl, so a misread request (say, a dropped «без молока») fails as well;
- an infeasible answer is checked too: every relaxation option is itself a recipe and must satisfy
  the request with that one constraint relaxed to the recommended value.
Prints a pass/fail table, writes eval/proof_report.md, exits 1 if anything fails.

    make proof                                  # in-process, real LLM from .env
    make proof ARGS="--url http://api:8000"     # the running compose service
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUESTS = ROOT / "eval" / "requests.jsonl"
REPORT = ROOT / "eval" / "proof_report.md"
NUTRIENTS = ("kcal", "protein_g", "fat_g", "carbs_g", "sugar_g", "fiber_g", "salt_g")
REDUCED_SUGAR_MAX_SHARE = 0.70  # Reg. (EC) 1924/2006 Annex: at least 30 % less sugar
MAX_INGREDIENTS = 6  # the technologist's rule: each ingredient is a supplier and an audit
LIMIT_TOL = 1e-6  # relative, a recomputed value vs a limit: float noise only
REPORTED_TOL = 1e-3  # a number reported by the service vs recomputed here (it rounds to 4 digits)
GRAM_TOL = 1e-5


# --- independent recomputation from the raw catalog -------------------------------------------


def load_yaml(name: str) -> dict[str, dict]:
    rows = yaml.safe_load((ROOT / "data" / name).read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


INGREDIENTS = load_yaml("ingredients.yaml")
TEMPLATES = load_yaml("templates.yaml")


@dataclass(frozen=True)
class Mix:
    per_100g: dict[str, float]
    sweetness: float
    cost: float  # UAH per kg
    allergens: frozenset[str]


def mix(grams: dict[str, float]) -> Mix:
    total = sum(grams.values())
    ings = {i: INGREDIENTS[i] for i in grams}
    per_100g = {
        k: sum(ings[i]["nutrients_per_100g"][k] * g for i, g in grams.items()) / total
        for k in NUTRIENTS
    }
    return Mix(
        per_100g=per_100g,
        sweetness=sum(ings[i]["sweetness_per_100g"] * g for i, g in grams.items()) / total,
        cost=sum(ings[i]["price_per_kg_uah"] * g for i, g in grams.items()) / total,
        allergens=frozenset(a for i in grams for a in ings[i].get("allergens", [])),
    )


def baseline(template_id: str) -> Mix:
    return mix(TEMPLATES[template_id]["base_recipe"])


def holds(op: str, actual: float, limit: float) -> bool:
    tol = LIMIT_TOL * max(1.0, abs(limit))
    return {
        ">=": actual >= limit - tol,
        "<=": actual <= limit + tol,
        "==": abs(actual - limit) <= tol,
    }[op]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= REPORTED_TOL * max(1.0, abs(b))


def num(value: float) -> str:
    """As the service writes numbers in explanations: 60.11 -> "60,11", 45.0 -> "45"."""
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")


# --- checks -----------------------------------------------------------------------------------


@dataclass
class Case:
    id: str
    expected: str
    got: str = "-"
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    summary: str = ""

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)


def check_recipe(case: Case, prefix: str, grams: dict[str, float], template_id: str, expect: dict):
    """The recipe itself against the template and the hand-written expectations."""
    unknown = sorted(set(grams) - set(INGREDIENTS))
    case.check(f"{prefix}ingredients known", not unknown, f"unknown: {unknown}" if unknown else "")
    if unknown:
        return None
    total = sum(grams.values())
    case.check(f"{prefix}grams sum to 1000", abs(total - 1000) <= GRAM_TOL, f"sum {total}")
    bounds = TEMPLATES[template_id]["category_bounds"]
    stray = sorted(i for i in grams if INGREDIENTS[i]["category"] not in bounds)
    case.check(
        f"{prefix}only template categories", not stray, f"not allowed: {stray}" if stray else ""
    )
    for cat, b in bounds.items():
        in_cat = sum(g for i, g in grams.items() if INGREDIENTS[i]["category"] == cat)
        ok = b["min_g"] - GRAM_TOL <= in_cat <= b["max_g"] + GRAM_TOL
        case.check(f"{prefix}{cat} {b['min_g']}–{b['max_g']} g", ok, f"{in_cat:.6f} g")

    limit = expect.get("max_ingredients", MAX_INGREDIENTS)
    case.check(f"{prefix}≤ {limit} ingredients", len(grams) <= limit, f"{len(grams)}")

    m, base = mix(grams), baseline(template_id)
    if free := expect.get("allergens_free"):
        present = sorted(m.allergens & set(free))
        case.check(
            f"{prefix}free of {', '.join(free)}",
            not present,
            f"contains {present}" if present else "",
        )
    if (protein := expect.get("protein_min_g")) is not None:
        limit = base.per_100g["protein_g"] if protein == "baseline" else float(protein)
        actual = m.per_100g["protein_g"]
        case.check(f"{prefix}protein ≥ {limit:.4f}", holds(">=", actual, limit), f"{actual:.4f}")
    if expect.get("reduced_sugar"):
        s_lim = base.per_100g["sugar_g"] * REDUCED_SUGAR_MAX_SHARE
        k_lim = base.per_100g["kcal"]
        s, k = m.per_100g["sugar_g"], m.per_100g["kcal"]
        case.check(
            f"{prefix}sugar ≤ 70 % of regular", holds("<=", s, s_lim), f"{s:.4f}/{s_lim:.4f}"
        )
        case.check(f"{prefix}kcal ≤ regular", holds("<=", k, k_lim), f"{k:.4f}/{k_lim:.4f}")
    if (pct := expect.get("sugar_reduction_pct")) is not None:
        s_lim = base.per_100g["sugar_g"] * (1 - pct / 100)
        s = m.per_100g["sugar_g"]
        case.check(f"{prefix}sugar ≥ {pct} % less", holds("<=", s, s_lim), f"{s:.4f}/{s_lim:.4f}")
    if (cost := expect.get("max_cost_uah_per_kg")) is not None:
        case.check(f"{prefix}cost ≤ {cost}", holds("<=", m.cost, cost), f"{m.cost:.4f}")
    return m


def recomputed_actual(name: str, grams: dict, m: Mix, template_id: str, expect: dict):
    if name == "total":
        return sum(grams.values())
    if name.startswith("category_"):
        cat = name.removeprefix("category_").rsplit("_", 1)[0]
        return sum(g for i, g in grams.items() if INGREDIENTS[i]["category"] == cat)
    simple = {
        "protein": m.per_100g["protein_g"],
        "sugar_reduced_sugar": m.per_100g["sugar_g"],
        "sugar_reduction": m.per_100g["sugar_g"],
        "sugar_reduced_kcal": m.per_100g["kcal"],
        "sweetness": m.sweetness,
        "cost": m.cost,
        "allergens_excluded": len(m.allergens & set(expect.get("allergens_free", []))),
        "max_ingredients": len(grams),
    }
    return simple.get(name)


def check_ok(case: Case, body: dict, expect: dict) -> None:
    template_id, grams = body["template"], body["recipe_grams"]
    case.check("template", template_id == expect["template"], f"got {template_id}")
    if template_id not in TEMPLATES:
        return
    m = check_recipe(case, "", grams, template_id, expect)
    if m is None:
        return
    case.summary = f"{m.cost:.2f} UAH/kg, protein {m.per_100g['protein_g']:.2f} g/100 g"

    reported = body["nutrition_per_100g"]
    off = [k for k in NUTRIENTS if not close(reported[k], m.per_100g[k])]
    case.check("reported nutrition = recomputed", not off, f"differs: {off}" if off else "")
    per_kg_off = [
        k for k in NUTRIENTS if not close(body["nutrition_per_kg"][k], m.per_100g[k] * 10)
    ]
    case.check(
        "per kg = 10 × per 100 g", not per_kg_off, f"differs: {per_kg_off}" if per_kg_off else ""
    )
    cost = body["cost_uah_per_kg"]
    case.check("reported cost = recomputed", close(cost, m.cost), f"{cost} vs {m.cost:.4f}")
    case.check(
        "reported allergens = recomputed",
        sorted(body["allergens"]) == sorted(m.allergens),
        f"{body['allergens']} vs {sorted(m.allergens)}",
    )
    if expect.get("reduced_sugar"):
        verified = body["claims_verified"].get("reduced_sugar")
        case.check("claim reported as verified", verified is True, f"got {verified}")

    for item in body["constraints_check"]:
        name = item["constraint"]
        actual = recomputed_actual(name, grams, m, template_id, expect)
        if actual is None:
            case.check(f"check «{name}» known", False, "unknown constraint name")
            continue
        ok = close(item["actual"], actual) and holds(item["op"], actual, item["required"])
        detail = (
            f"recomputed {actual:.4f} {item['op']} {item['required']} (reported {item['actual']})"
        )
        case.check(f"check «{name}» recomputed", ok, detail)


def check_infeasible(case: Case, body: dict, expect: dict) -> None:
    case.check("template", body["template"] == expect["template"], f"got {body['template']}")
    case.check("explanation given", bool(body["explanation"].strip()))
    options = body["relaxation_options"]
    case.check("relaxation or reason given", bool(options or body["blocking_reasons"]))
    if want := expect.get("first_relaxation"):
        first = options[0]["constraint"] if options else None
        case.check(f"first relaxation is {want}", first == want, f"got {first}")
    for n, opt in enumerate(options, start=1):
        relaxed, name = dict(expect), opt["constraint"]
        if name == "protein_g_per_100g":
            relaxed["protein_min_g"] = opt["minimal_feasible"]
            case.check(f"#{n} protein lowered", opt["minimal_feasible"] < opt["requested"])
        elif name == "sugar_reduction_pct":
            relaxed["sugar_reduction_pct"] = opt["minimal_feasible"]
            case.check(f"#{n} reduction lowered", opt["minimal_feasible"] < opt["requested"])
        elif name == "cost_ceiling_uah_per_kg":
            relaxed["max_cost_uah_per_kg"] = opt["minimal_feasible"]
            case.check(f"#{n} ceiling raised", opt["minimal_feasible"] > opt["requested"])
        elif name == "sugar_reduced_claim":
            relaxed["reduced_sugar"] = False
        elif name == "max_ingredients":
            relaxed["max_ingredients"] = opt["minimal_feasible"]
            case.check(f"#{n} limit raised", opt["minimal_feasible"] > opt["requested"])
        else:
            case.check(f"#{n} known relaxation", False, name)
            continue
        m = check_recipe(case, f"#{n} ", opt["recipe_grams"], body["template"], relaxed)
        if m is not None:
            ok = close(opt["cost_uah_per_kg"], m.cost)
            case.check(f"#{n} reported cost = recomputed", ok, f"{opt['cost_uah_per_kg']}")
        if name != "sugar_reduced_claim":
            value = num(opt["minimal_feasible"])
            case.check(f"#{n} explanation cites {value}", value in body["explanation"])
    if options:
        o = options[0]
        case.summary = f"{o['constraint']}: {o['requested']} → {o['minimal_feasible']}"


def check_unsupported(case: Case, body: dict) -> None:
    case.check("explanation given", bool(body["explanation"].strip()))
    case.check("lists supported", set(body["supported_templates"]) == set(TEMPLATES))
    case.summary = body["explanation"][:60]


# --- transport --------------------------------------------------------------------------------


class _NoRuns:
    async def save(self, run) -> None:
        return None


async def make_client(url: str | None) -> tuple[httpx.AsyncClient, str]:
    if url:
        return httpx.AsyncClient(base_url=url, timeout=180), f"server {url}"
    from app.config import settings
    from app.llm import create_llm_client
    from app.main import app
    from app.seed import load_catalog

    app.state.catalog = load_catalog()
    app.state.llm = create_llm_client(settings)
    app.state.runs = _NoRuns()
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://proof", timeout=180)
    return client, f"in-process, LLM {app.state.llm.provider}"


async def run_case(client: httpx.AsyncClient, row: dict) -> Case:
    expect = row["expect"]
    case = Case(row["id"], expect["status"])
    try:
        response = await client.post("/recipe", json={"request": row["request"]})
    except httpx.HTTPError as exc:
        case.check("service answered", False, f"{type(exc).__name__}: {exc}")
        return case
    body = response.json()
    case.got = body.get("status") or f"HTTP {response.status_code}"
    if response.status_code != 200:
        case.check("service answered", False, json.dumps(body, ensure_ascii=False)[:200])
        return case
    case.check("status", case.got == expect["status"], f"got {case.got}")
    if case.got != expect["status"]:
        return case
    if case.got == "ok":
        check_ok(case, body, expect)
    elif case.got == "infeasible":
        check_infeasible(case, body, expect)
    else:
        check_unsupported(case, body)
    return case


# --- output -----------------------------------------------------------------------------------


def table(cases: list[Case]) -> list[str]:
    rows = [
        (c.id, c.expected, c.got, f"{sum(ok for _, ok, _ in c.checks)}/{len(c.checks)}",
         "PASS" if c.passed else "FAIL", c.summary)
        for c in cases
    ]  # fmt: skip
    head = ("request", "expected", "got", "checks", "result", "summary")
    return [
        "| " + " | ".join(head) + " |",
        "|" + "|".join("---" for _ in head) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def failures(cases: list[Case]) -> list[str]:
    return [
        f"- {c.id}: {name} — {detail}" for c in cases for name, ok, detail in c.checks if not ok
    ]


def write_report(cases: list[Case], mode: str) -> None:
    passed = sum(c.passed for c in cases)
    lines = [
        "# Proof report",
        "",
        f"Generated by `make proof` ({mode}) on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC: "
        f"**{passed}/{len(cases)} requests pass**.",
        "",
        "Every number is recomputed by `eval/proof.py` from `data/*.yaml`, independently of the "
        "solver; `satisfied` from the response is never trusted. Expectations are written by hand "
        "in `eval/requests.jsonl`, not taken from the LLM's reading of the request.",
        "",
        *table(cases),
        "",
        "## Checks per request",
    ]
    for c in cases:
        lines += ["", f"### {c.id} — {'PASS' if c.passed else 'FAIL'}", ""]
        lines += [
            f"- {'✅' if ok else '❌'} {name}" + (f" ({d})" if d else "")
            for name, ok, d in c.checks
        ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", help="a running server, e.g. http://api:8000 (default: in-process)"
    )
    parser.add_argument("--only", nargs="*", help="request ids to run")
    args = parser.parse_args()

    rows = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line]
    if args.only:
        rows = [r for r in rows if r["id"] in args.only]
    client, mode = await make_client(args.url)
    async with client:
        cases = [await run_case(client, row) for row in rows]  # one by one: free LLM quotas

    print(f"Proof: {mode}\n")
    print("\n".join(table(cases)))
    if bad := failures(cases):
        print("\nFailed checks:\n" + "\n".join(bad))
    summary = f"\n{sum(c.passed for c in cases)}/{len(cases)} pass"
    if args.only:  # a partial run must not overwrite the full report
        print(summary)
    else:
        write_report(cases, mode)
        print(f"{summary}; report: {REPORT.relative_to(ROOT)}")
    return 0 if all(c.passed for c in cases) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
