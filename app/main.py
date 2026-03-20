"""
RobustStat API
==============
Automated statistical stability testing.
Upload a CSV, get a PASS/FAIL verdict + full report.

Endpoints:
  GET  /              — API info
  GET  /health        — health check
  POST /analyze       — run dual-channel stability test
  GET  /demo          — run a built-in demo and return sample report
"""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.stability import StabilityInput, StabilityReport, run_stability

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RobustStat API",
    description=(
        "Stop guessing convergence. Upload a CSV, get a PASS/FAIL verdict "
        "backed by Theil-Sen slope + Bootstrap CI + S1/S2 dual-channel analysis. "
        "Physics-agnostic: works for ML training runs, simulations, time-series, "
        "numerical solvers — anything that should converge."
    ),
    version="1.0.0",
    contact={
        "name": "RobustStat",
        "url": "https://github.com/mrvoid1701-code/RobustStat-API",
    },
    license_info={"name": "MIT"},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=422, detail="CSV file is empty or has no data rows.")
    return rows


def _report_to_dict(r: StabilityReport) -> dict[str, Any]:
    return {
        "decision": r.decision,
        "summary": r.summary,
        "checks": r.checks,
        "s2_structural": {
            "pass_fraction": round(r.s2.pass_fraction, 6),
            "group_results": r.s2.group_results,
        },
        "s1_energetic": {
            "full_slope_median": r.s1.full_slope_median,
            "full_slope_ci_low": r.s1.full_slope_ci_low,
            "full_slope_ci_high": r.s1.full_slope_ci_high,
            "bulk_slope_median": r.s1.bulk_slope_median,
            "bulk_slope_ci_low": r.s1.bulk_slope_ci_low,
            "bulk_slope_ci_high": r.s1.bulk_slope_ci_high,
            "bulk_valid_fraction": round(r.s1.bulk_valid_fraction, 6),
            "slopes_per_group": {
                k: round(v, 9) for k, v in r.s1.full_slopes_per_group.items()
            },
        },
        "meta": {
            "group_count": r.group_count,
            "scale_levels": r.scale_levels,
            "bulk_levels": r.bulk_levels,
            "bootstrap_reps": r.bootstrap_reps,
            "ci_alpha": r.ci_alpha,
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "interpretation": _interpret(r),
    }


def _interpret(r: StabilityReport) -> dict[str, str]:
    """Plain-English explanation of each check result."""
    ci_lo = r.s1.full_slope_ci_low
    ci_hi = r.s1.full_slope_ci_high
    conf = int((1 - r.ci_alpha) * 100)

    if r.decision == "PASS":
        verdict_text = (
            "Your metric is statistically confirmed to be converging. "
            f"The {conf}% CI for the median Theil-Sen slope is "
            f"[{ci_lo:.4f}, {ci_hi:.4f}], entirely below zero — "
            "meaning the metric is reliably decreasing as scale increases."
        )
    else:
        if ci_hi >= 0:
            verdict_text = (
                f"Convergence NOT confirmed. The {conf}% CI for the slope is "
                f"[{ci_lo:.4f}, {ci_hi:.4f}], which includes zero or positive values. "
                "The metric does not reliably decrease with scale."
            )
        else:
            verdict_text = (
                "One or more structural (S2) gates failed — "
                "check the 's2_structural.group_results' field for details."
            )

    checks_text = {}
    for k, v in r.checks.items():
        if k == "s2_all_groups_pass":
            checks_text[k] = (
                "All groups passed structural gates." if v
                else f"Some groups failed structural gates (pass_fraction={r.s2.pass_fraction:.2f})."
            )
        elif k == "bulk_valid_fraction_ok":
            checks_text[k] = (
                "Enough groups had valid bulk-scale data." if v
                else f"Too few groups had valid bulk data ({r.s1.bulk_valid_fraction:.2f} fraction)."
            )
        elif k == "s1_full_slope_ci_negative":
            checks_text[k] = (
                "Full-scale trend: CI upper bound is negative → converging." if v
                else "Full-scale trend: CI upper bound ≥ 0 → not confirmed converging."
            )
        elif k == "s1_bulk_slope_ci_negative":
            checks_text[k] = (
                "Bulk-scale trend: CI upper bound is negative → converging." if v
                else "Bulk-scale trend: CI upper bound ≥ 0 → not confirmed converging."
            )

    return {"verdict": verdict_text, "checks": checks_text}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "RobustStat API",
        "version": "1.0.0",
        "description": (
            "Upload a CSV with convergence data → get PASS/FAIL + "
            "Theil-Sen slope, Bootstrap CI, S1/S2 dual-channel stability report."
        ),
        "endpoints": {
            "POST /analyze": "Run stability test on your CSV",
            "GET  /demo":    "Try a built-in convergence demo",
            "GET  /health":  "Health check",
        },
        "csv_format": {
            "required_columns": ["group", "scale", "metric"],
            "optional_columns": ["any boolean gate columns (pass/fail)"],
            "example_row": {"group": "run_1", "scale": "100", "metric": "0.042"},
        },
    }


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.post(
    "/analyze",
    tags=["Analysis"],
    summary="Run dual-channel stability test",
    response_description="PASS/FAIL verdict with full statistical report",
)
async def analyze(
    file: Annotated[UploadFile, File(description="CSV file with convergence data")],
    group_col: Annotated[str, Form()] = "group",
    scale_col: Annotated[str, Form()] = "scale",
    metric_col: Annotated[str, Form()] = "metric",
    gate_cols: Annotated[str, Form(description="Comma-separated boolean gate column names (optional)")] = "",
    bulk_flag_cols: Annotated[str, Form(description="Comma-separated bulk eligibility flag columns (optional)")] = "",
    min_eligible_per_level: Annotated[int, Form(ge=1, le=100)] = 3,
    bulk_min_fraction: Annotated[float, Form(ge=0.0, le=1.0)] = 0.80,
    bootstrap_reps: Annotated[int, Form(ge=100, le=10000)] = 2000,
    ci_alpha: Annotated[float, Form(ge=0.01, le=0.20)] = 0.05,
):
    """
    ## Run a full stability test on your convergence data.

    ### CSV Format
    Your CSV must have at minimum three columns:
    - **group** — identifier for each independent run/trial/seed
    - **scale** — numeric scale/size parameter (e.g., number of nodes, epochs, iterations)
    - **metric** — the convergence metric you're testing (lower = better assumed)

    You can rename these columns via the form parameters.

    ### Optional: Structural Gate Columns
    Add boolean columns (`pass`/`fail` or `true`/`false` or `1`/`0`) for hard
    structural checks (S2 lane). Pass their names via `gate_cols`.

    ### How it works
    1. **S2 Structural lane** — deterministic: every gate column must pass for every row in a group
    2. **S1 Energetic lane** — statistical: Theil-Sen slope + Bootstrap CI must be entirely negative
       (metric decreasing with scale → converging)

    ### Decision
    - **PASS** — all checks pass: metric is statistically confirmed converging
    - **FAIL** — one or more checks failed
    """

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    rows = _parse_csv(content)

    # Validate columns exist
    first_row = rows[0]
    for col_name, col_val in [
        ("group_col", group_col),
        ("scale_col", scale_col),
        ("metric_col", metric_col),
    ]:
        if col_val not in first_row:
            available = list(first_row.keys())
            raise HTTPException(
                status_code=422,
                detail=f"Column '{col_val}' not found. Available columns: {available}",
            )

    gate_list = [c.strip() for c in gate_cols.split(",") if c.strip()]
    bulk_list  = [c.strip() for c in bulk_flag_cols.split(",") if c.strip()]

    for col in gate_list + bulk_list:
        if col not in first_row:
            raise HTTPException(
                status_code=422,
                detail=f"Gate column '{col}' not found. Available columns: {list(first_row.keys())}",
            )

    inp = StabilityInput(
        rows=rows,
        group_col=group_col,
        scale_col=scale_col,
        metric_col=metric_col,
        gate_cols=gate_list,
        bulk_flag_cols=bulk_list,
        min_eligible_per_level=min_eligible_per_level,
        bulk_min_fraction=bulk_min_fraction,
        bootstrap_reps=bootstrap_reps,
        ci_alpha=ci_alpha,
    )

    try:
        report = run_stability(inp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")

    return JSONResponse(content=_report_to_dict(report))


@app.get(
    "/demo",
    tags=["Analysis"],
    summary="Run a built-in convergence demo",
)
def demo(
    scenario: Annotated[str, "converging | diverging | noisy"] = "converging",
):
    """
    ## Try RobustStat without uploading a file.

    Generates synthetic convergence data and runs the full analysis.

    ### Scenarios
    - **converging** — metric clearly decreases with scale (should PASS)
    - **diverging**  — metric increases with scale (should FAIL)
    - **noisy**      — metric decreases but with high noise (borderline)

    Pass `?scenario=diverging` or `?scenario=noisy` to try other cases.
    """
    import random as _rng

    rng = _rng.Random(42)
    rows = []
    scales = [50, 100, 200, 400, 800]
    groups = [f"run_{i}" for i in range(1, 9)]

    for g in groups:
        base = rng.uniform(0.08, 0.15)
        for s in scales:
            for _ in range(4):  # 4 profiles per (group, scale)
                if scenario == "converging":
                    m = base / math.sqrt(s / 50) + rng.gauss(0, 0.002)
                elif scenario == "diverging":
                    m = base * (s / 50) ** 0.3 + rng.gauss(0, 0.002)
                else:  # noisy
                    m = base / (s / 50) ** 0.2 + rng.gauss(0, base * 0.4)

                rows.append({
                    "group": g,
                    "scale": str(s),
                    "metric": f"{max(m, 0.0):.6f}",
                    "gate_positive": "pass" if m >= 0 else "fail",
                })

    inp = StabilityInput(
        rows=rows,
        group_col="group",
        scale_col="scale",
        metric_col="metric",
        gate_cols=["gate_positive"],
        bootstrap_reps=1000,
        ci_alpha=0.05,
    )

    report = run_stability(inp)
    result = _report_to_dict(report)
    result["demo_scenario"] = scenario
    result["demo_note"] = (
        f"Synthetic data: {len(groups)} groups × {len(scales)} scales × 4 profiles. "
        f"Scenario '{scenario}' → expected {'PASS' if scenario == 'converging' else 'FAIL'}."
    )
    return JSONResponse(content=result)
