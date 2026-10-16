#!/usr/bin/env python3
"""SAD-OTR-TM ablation (final-check revision). N=32, K=4, delta=15 ms, seeds 0-9.

Variants: slice cap L_max, amortization (best-prefix) rule off, lazy stopping
off (exactness check of Prop. tm-dom), first-slice policy, candidate-duration
set size, and a wall-clock decision budget.  Run serially (single process) so
latencies are not inflated by concurrent workers.
Output: results/tm_ablation_seeds.csv, results/tm_ablation.csv, results/tm_ablation.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sad_otr.demands import make_phase_sequence  # noqa: E402
from sad_otr.baselines import eps_time  # noqa: E402
from sad_otr.multislot import sad_otr_tm_variant  # noqa: E402

OUT = ROOT / "results"
N, K, BO, BE, DELTA = 32, 4, 800.0, 400.0, 0.015

VARIANTS = [("default (L_max=32)", {})]
VARIANTS += [(f"L_max={L}", {"n_slices": L}) for L in (1, 2, 3, 4, 6, 8, 16)]
VARIANTS += [
    ("no lazy stop", {"lazy_stop": False}),
    ("amortization off (adopt all slices)", {"adopt_all": True}),
    ("first slice always charged", {"first_slice": "ghem_charged"}),
    ("no GHEM-AC first slice", {"first_slice": "greedy"}),
    ("|A|<=4", {"max_alphas": 4}),
    ("|A|<=16", {"max_alphas": 16}),
    ("|A| unrestricted", {"max_alphas": 10**9}),
]
VARIANTS += [(f"budget {b} ms", {"time_budget_s": b / 1000}) for b in (5, 10, 20, 40)]


def ci(v):
    v = np.asarray(v, float); n = len(v)
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))


def main():
    rows = []
    for seed in range(10):
        phases = make_phase_sequence(N, 12, seed)
        eps = float(np.mean([eps_time(D, BE) for _, D in phases]))
        for name, kw in VARIANTS:
            E = np.zeros((N, N), dtype=np.int8)
            Ts, nrec, lat, ob, tb = [], 0, [], 0.0, 0.0
            for _, D in phases:
                t0 = time.perf_counter()
                o = sad_otr_tm_variant(D, E, K, BO, BE, DELTA, **kw)
                lat.append(time.perf_counter() - t0)
                Ts.append(o["T"]); nrec += o["n_charged"]; ob += o["opt_bytes"]; tb += o["total_bytes"]
                E = o["E_last"]
            rows.append({"seed": seed, "variant": name, "mean_phase_s": float(np.mean(Ts)),
                         "speedup_vs_eps": eps / float(np.mean(Ts)), "n_reconfig": nrec,
                         "optical_pct": 100 * ob / tb, "mean_latency_ms": 1000 * float(np.mean(lat)),
                         "max_latency_ms": 1000 * float(np.max(lat))})
        print("seed", seed, "done", flush=True)
    with (OUT / "tm_ablation_seeds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    base = {r["seed"]: r["mean_phase_s"] for r in rows if r["variant"] == VARIANTS[0][0]}
    summ = []
    for name, _ in VARIANTS:
        sub = [r for r in rows if r["variant"] == name]
        m, h = ci([r["mean_phase_s"] for r in sub])
        rm, rh = ci([r["mean_phase_s"] / base[r["seed"]] for r in sub])
        s = {"variant": name, "mean_phase_s": m, "ci95": h, "ratio_to_default": rm, "ratio_ci95": rh,
             "speedup_vs_eps": ci([r["speedup_vs_eps"] for r in sub])[0],
             "n_reconfig": ci([r["n_reconfig"] for r in sub])[0],
             "optical_pct": ci([r["optical_pct"] for r in sub])[0],
             "mean_latency_ms": ci([r["mean_latency_ms"] for r in sub])[0],
             "max_latency_ms": float(max(r["max_latency_ms"] for r in sub))}
        summ.append(s)
        print(f"{name:40s} T={m:.4f}±{h:.4f} ratio={rm:.4f}±{rh:.4f} rec={s['n_reconfig']:.1f} "
              f"lat={s['mean_latency_ms']:.1f} ms (max {s['max_latency_ms']:.0f})")
    with (OUT / "tm_ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    (OUT / "tm_ablation.json").write_text(json.dumps({"params": {"N": N, "K": K, "B_opt": BO, "B_eps": BE,
                                                      "delta_s": DELTA, "seeds": list(range(10))},
                                                      "summary": summ}, indent=2))


if __name__ == "__main__":
    main()
