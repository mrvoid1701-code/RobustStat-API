# RobustStat API

**Automated Statistical Stability Testing**

Stop guessing convergence. Upload a CSV, get a `PASS` / `FAIL` verdict backed by mathematical rigor.

---

## What it does

RobustStat uses a **dual-channel engine** (ported and generalized from the QNG Workspace physics validation framework) to test whether your convergence metric is statistically confirmed to decrease as scale increases.

| Channel | Method | What it checks |
|---------|--------|----------------|
| **S2 Structural** | Hard deterministic gates | Every structural flag must pass for every row in every group |
| **S1 Energetic** | Theil-Sen slope + Bootstrap CI | The median slope CI upper bound must be < 0 (converging) |

**Physics-agnostic** — works for ML training loss, numerical solver residuals, simulation energy drift, time-series stabilization, anything that should converge.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info and CSV format guide |
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Run stability test on your CSV |
| `GET` | `/demo` | Try with built-in synthetic data |

### POST `/analyze`

Upload a CSV with at least three columns:

```
group,scale,metric
run_1,100,0.0842
run_1,200,0.0611
run_1,400,0.0443
run_2,100,0.0901
...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file` | — | CSV file (required) |
| `group_col` | `group` | Column identifying each run/trial/seed |
| `scale_col` | `scale` | Numeric scale/size column |
| `metric_col` | `metric` | The convergence metric (lower = better) |
| `gate_cols` | `""` | Comma-separated boolean gate columns for S2 lane |
| `bootstrap_reps` | `2000` | Bootstrap resampling count |
| `ci_alpha` | `0.05` | Two-sided CI alpha (0.05 → 95% CI) |

### Example response

```json
{
  "decision": "PASS",
  "summary": "PASS — 4/4 checks passed | full_slope_CI=[-0.0003,-0.0001] | groups=8 | scales=5",
  "checks": {
    "s2_all_groups_pass": true,
    "bulk_valid_fraction_ok": true,
    "s1_full_slope_ci_negative": true,
    "s1_bulk_slope_ci_negative": true
  },
  "s1_energetic": {
    "full_slope_median": -0.000189,
    "full_slope_ci_low": -0.000312,
    "full_slope_ci_high": -0.000098
  },
  "interpretation": {
    "verdict": "Your metric is statistically confirmed to be converging...",
    "checks": {}
  }
}
```

---

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open interactive docs at `http://localhost:8000/docs`

## Run with Docker

```bash
docker build -t robuststat .
docker run -p 8000:8000 robuststat
```

---

## Deploy to production

Designed for one-command deploy on **Railway**, **Render**, **Fly.io**, or **AWS Lambda**.

---

## License

MIT
