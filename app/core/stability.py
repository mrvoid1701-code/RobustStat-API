"""
RobustStat Core Engine
======================
Physics-agnostic dual-channel stability testing.

Channels:
  S2 (Structural) — hard deterministic pass/fail gates per row/group
  S1 (Energetic)  — statistical convergence trend via Theil-Sen slope + Bootstrap CI

Ported and generalized from QNG Workspace stability convergence gate v6.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core math primitives
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def theil_sen_slope(xs: list[float], ys: list[float]) -> float:
    """Robust non-parametric slope estimator (Theil-Sen)."""
    slopes: list[float] = []
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            dx = xs[j] - xs[i]
            if abs(dx) <= 1e-18:
                continue
            slopes.append((ys[j] - ys[i]) / dx)
    return _median(slopes) if slopes else 0.0


def bootstrap_median_ci(
    vals: list[float],
    reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 1701,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for the median.

    Returns (median, ci_lower, ci_upper).
    """
    if not vals:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    boots: list[float] = [
        _median([vals[rng.randrange(n)] for _ in range(n)])
        for _ in range(max(100, reps))
    ]
    return (
        _median(vals),
        _percentile(boots, alpha / 2.0),
        _percentile(boots, 1.0 - alpha / 2.0),
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StabilityInput:
    """
    Parsed + validated input for the stability engine.

    rows       : list of dicts, each row from the uploaded CSV
    group_col  : column identifying each independent run/seed/trial
    scale_col  : column identifying scale/size (numeric, for slope computation)
    metric_col : the convergence metric to test (lower-is-better assumed)
    gate_cols  : list of boolean columns for S2 structural gates
                 (values accepted: 'pass'/'true'/'1'/'yes' → pass, else fail)
    bulk_flag_cols : extra boolean columns required for a row to be
                     eligible for the bulk (inner-scale) trend analysis
    min_eligible_per_level : minimum eligible rows per scale level
    bulk_min_fraction      : minimum fraction of groups that must be bulk-valid
    bootstrap_reps         : bootstrap resampling count
    ci_alpha               : two-sided CI alpha (default 0.05 → 95% CI)
    """
    rows: list[dict[str, Any]]
    group_col: str = "group"
    scale_col: str = "scale"
    metric_col: str = "metric"
    gate_cols: list[str] = field(default_factory=list)
    bulk_flag_cols: list[str] = field(default_factory=list)
    min_eligible_per_level: int = 3
    bulk_min_fraction: float = 0.80
    bootstrap_reps: int = 2000
    ci_alpha: float = 0.05


@dataclass
class S2Result:
    pass_fraction: float
    group_results: dict[str, bool]  # group → all_gates_pass


@dataclass
class S1Result:
    full_slope_median: float
    full_slope_ci_low: float
    full_slope_ci_high: float
    bulk_slope_median: float
    bulk_slope_ci_low: float
    bulk_slope_ci_high: float
    full_slopes_per_group: dict[str, float]
    bulk_valid_fraction: float


@dataclass
class StabilityReport:
    decision: str                  # "PASS" or "FAIL"
    checks: dict[str, bool]
    s2: S2Result
    s1: S1Result
    group_count: int
    scale_levels: list[float]
    bulk_levels: list[float]
    bootstrap_reps: int
    ci_alpha: float
    summary: str                   # human-readable one-liner


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _is_pass(v: Any) -> bool:
    return str(v or "").strip().lower() in {"pass", "true", "1", "yes"}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(str(v).strip())
        return default if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return default


def run_stability(inp: StabilityInput) -> StabilityReport:
    """
    Run the full dual-channel stability test and return a StabilityReport.
    """

    # ── 0. Group + index rows ─────────────────────────────────────────────
    groups: dict[str, list[dict]] = {}
    all_scales: set[float] = set()

    for row in inp.rows:
        g = str(row.get(inp.group_col, "")).strip()
        if not g:
            continue
        s = _to_float(row.get(inp.scale_col, 0))
        all_scales.add(s)
        groups.setdefault(g, []).append(row)

    scales = sorted(all_scales)
    # bulk = inner levels (exclude first and last — boundary effects)
    bulk_levels = scales[1:-1] if len(scales) > 2 else scales

    # ── 1. S2 Structural lane ─────────────────────────────────────────────
    s2_group_pass: dict[str, bool] = {}
    for g, rows in groups.items():
        all_pass = all(
            all(_is_pass(row.get(gc, "")) for gc in inp.gate_cols)
            for row in rows
        ) if inp.gate_cols else True
        s2_group_pass[g] = all_pass

    s2_pass_fraction = (
        sum(s2_group_pass.values()) / len(s2_group_pass)
        if s2_group_pass else 0.0
    )

    # ── 2. S1 Energetic lane ──────────────────────────────────────────────
    full_slopes: list[float] = []
    bulk_slopes: list[float] = []
    full_slopes_per_group: dict[str, float] = {}
    bulk_valid_count = 0

    for g, rows in groups.items():
        # Per-scale median metric (full)
        by_scale: dict[float, list[float]] = {}
        by_scale_bulk: dict[float, list[float]] = {}

        for row in rows:
            s = _to_float(row.get(inp.scale_col, 0))
            m = _to_float(row.get(inp.metric_col, 0))
            by_scale.setdefault(s, []).append(m)

            # Bulk eligibility
            bulk_ok = all(_is_pass(row.get(fc, "")) for fc in inp.bulk_flag_cols) \
                if inp.bulk_flag_cols else True
            if bulk_ok and s in bulk_levels:
                by_scale_bulk.setdefault(s, []).append(m)

        full_xs = sorted(by_scale.keys())
        full_ys = [_median(by_scale[s]) for s in full_xs]
        sl = theil_sen_slope(full_xs, full_ys) if len(full_xs) >= 2 else 0.0
        full_slopes.append(sl)
        full_slopes_per_group[g] = sl

        # Bulk trend
        eligible_levels = [
            lv for lv in bulk_levels
            if lv in by_scale_bulk
            and len(by_scale_bulk[lv]) >= inp.min_eligible_per_level
        ]
        if len(eligible_levels) >= 2:
            bx = eligible_levels
            by = [_median(by_scale_bulk[lv]) for lv in bx]
            bulk_slopes.append(theil_sen_slope(bx, by))
            bulk_valid_count += 1

    bulk_valid_fraction = bulk_valid_count / max(len(groups), 1)

    # Bootstrap CIs for median slope
    full_med, full_ci_lo, full_ci_hi = bootstrap_median_ci(
        full_slopes, reps=inp.bootstrap_reps, alpha=inp.ci_alpha, seed=1701
    )
    bulk_med, bulk_ci_lo, bulk_ci_hi = bootstrap_median_ci(
        bulk_slopes, reps=inp.bootstrap_reps, alpha=inp.ci_alpha, seed=1702
    ) if bulk_slopes else (0.0, 0.0, 0.0)

    # ── 3. Decision ───────────────────────────────────────────────────────
    checks = {
        "s2_all_groups_pass":             abs(s2_pass_fraction - 1.0) <= 1e-9,
        "bulk_valid_fraction_ok":         bulk_valid_fraction >= inp.bulk_min_fraction,
        "s1_full_slope_ci_negative":      full_ci_hi < 0.0,
        "s1_bulk_slope_ci_negative":      bulk_ci_hi < 0.0 if bulk_slopes else False,
    }

    # If no gate_cols supplied, skip S2 check
    if not inp.gate_cols:
        checks["s2_all_groups_pass"] = True

    # If no bulk levels exist, skip bulk CI check
    if len(scales) <= 2:
        checks["bulk_valid_fraction_ok"] = True
        checks["s1_bulk_slope_ci_negative"] = checks["s1_full_slope_ci_negative"]

    decision = "PASS" if all(checks.values()) else "FAIL"

    passed = sum(checks.values())
    total = len(checks)
    summary = (
        f"{decision} — {passed}/{total} checks passed | "
        f"full_slope_CI=[{full_ci_lo:.4f},{full_ci_hi:.4f}] | "
        f"groups={len(groups)} | scales={len(scales)}"
    )

    return StabilityReport(
        decision=decision,
        checks=checks,
        s2=S2Result(
            pass_fraction=s2_pass_fraction,
            group_results=s2_group_pass,
        ),
        s1=S1Result(
            full_slope_median=full_med,
            full_slope_ci_low=full_ci_lo,
            full_slope_ci_high=full_ci_hi,
            bulk_slope_median=bulk_med,
            bulk_slope_ci_low=bulk_ci_lo,
            bulk_slope_ci_high=bulk_ci_hi,
            full_slopes_per_group=full_slopes_per_group,
            bulk_valid_fraction=bulk_valid_fraction,
        ),
        group_count=len(groups),
        scale_levels=scales,
        bulk_levels=bulk_levels,
        bootstrap_reps=inp.bootstrap_reps,
        ci_alpha=inp.ci_alpha,
        summary=summary,
    )
