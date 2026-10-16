#!/usr/bin/env python3
"""Paired significance tests on the 10-seed baseline study (final-check revision).

Unit of analysis: the seed (mean phase time over 12 phases), n = 10, paired.
Tests: two-sided paired t-test on log(time) (ratio scale) and two-sided exact
Wilcoxon signed-rank test on the paired differences.  Effect sizes: mean paired
ratio with 95% t-CI, Cohen's d_z on log ratios, and matched-pairs rank-biserial
correlation r = (W+ - W-)/(W+ + W-).  Holm correction over the family of
comparisons.  Reads results/baselines_seeds.csv only.
Output: results/significance.csv, results/significance.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

PAIRS = [
    ("SAD-OTR-TM", "Eclipse-K+LP"),
    ("SAD-OTR-TM", "Eclipse-K"),
    ("SAD-OTR-TM", "SAD-OTR (GHEM-AC)"),
    ("SAD-OTR-TM", "Solstice-K+LP"),
    ("SAD-OTR (GHEM-AC)", "MWM per phase"),
    ("SAD-OTR (GHEM-AC)", "RotorNet-style"),
    ("SAD-OTR (GHEM-AC)", "Static OCS"),
    ("SAD-OTR (GHEM-AC)", "Solstice-K"),
    ("Eclipse-K", "SAD-OTR (GHEM-AC)"),
]


def main() -> None:
    rows = list(csv.DictReader((OUT / "baselines_seeds.csv").open()))
    by = {}
    for r in rows:
        by.setdefault(r["scheme"], {})[int(r["seed"])] = float(r["mean_phase_s"])
    seeds = sorted(by["EPS Clos"])
    res = []
    for a, b in PAIRS:
        x = np.array([by[a][s] for s in seeds]); y = np.array([by[b][s] for s in seeds])
        lr = np.log(x / y)
        n = len(lr)
        t = stats.ttest_rel(np.log(x), np.log(y))
        w = stats.wilcoxon(x, y, alternative="two-sided", method="exact")
        d = x - y
        ranks = stats.rankdata(np.abs(d))
        wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
        rbc = (wp - wm) / (wp + wm)
        ratio = x / y
        h = stats.t.ppf(0.975, n - 1) * ratio.std(ddof=1) / np.sqrt(n)
        res.append({
            "a": a, "b": b, "n": n,
            "mean_ratio_a_over_b": float(ratio.mean()), "ratio_ci95": float(h),
            "wins_a_faster": int((x < y).sum()),
            "t_stat": float(t.statistic), "p_t": float(t.pvalue),
            "wilcoxon_W": float(w.statistic), "p_wilcoxon": float(w.pvalue),
            "cohen_dz_log": float(lr.mean() / lr.std(ddof=1)),
            "rank_biserial": float(rbc),
        })
    # Holm on Wilcoxon and t p-values
    for key in ("p_t", "p_wilcoxon"):
        order = np.argsort([r[key] for r in res])
        m = len(res); run = 0.0
        for i, idx in enumerate(order):
            adj = min(1.0, (m - i) * res[idx][key]); run = max(run, adj)
            res[idx][key + "_holm"] = run
    with (OUT / "significance.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(res[0].keys())); wr.writeheader(); wr.writerows(res)
    (OUT / "significance.json").write_text(json.dumps({"unit": "seed (n=10)", "tests": res}, indent=2))
    for r in res:
        print(f"{r['a']} vs {r['b']}: ratio {r['mean_ratio_a_over_b']:.4f}±{r['ratio_ci95']:.4f} "
              f"wins {r['wins_a_faster']}/10 p_t={r['p_t']:.2e} (holm {r['p_t_holm']:.2e}) "
              f"p_W={r['p_wilcoxon']:.4f} (holm {r['p_wilcoxon_holm']:.4f}) dz={r['cohen_dz_log']:.2f} rbc={r['rank_biserial']:.2f}")


if __name__ == "__main__":
    main()
