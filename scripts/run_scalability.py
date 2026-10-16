#!/usr/bin/env python3
"""Fabric-size sweep (Q1 revision): runtime and quality of GHEM-AC vs the exact
MILP, plus SAD-OTR-TM runtime and distance to the ideal bound.

Families: 'uniform' (primary suite, EP fan-out 4) and 'skewed' (EP fan-out 8,
Zipf(1.0) expert popularity).  N in {16, 32, 64, 128, 256}; seeds 0-4
(0-2 at N=256).  CBC: gapRel=0, threads=1, timeLimit=60 s.  Worker processes run
in parallel (4 workers on an 8-core box); latencies are therefore measured
under moderate contention and should be read as upper bounds.

Outputs: results/scalability_raw.csv, results/scalability.csv,
         results/scalability.json, results/run_scalability.log
"""

from __future__ import annotations

import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.baselines import ideal_lower_bound  # noqa: E402
from sad_otr.closed_form import phase_time  # noqa: E402
from sad_otr.demands import make_phase_sequence, make_phase_sequence_skewed  # noqa: E402
from sad_otr.ghem_ac import ghem_ac  # noqa: E402
from sad_otr.milp import solve_exact  # noqa: E402
from sad_otr.multislot import sad_otr_tm  # noqa: E402

OUT = ROOT / "results"
K, BO, BE, DELTA = 4, 800.0, 400.0, 0.015
TL = 60.0


def job(args):
    family, N, seed = args
    ph = (make_phase_sequence(N, 12, seed) if family == "uniform"
          else make_phase_sequence_skewed(N, 12, seed, fanout=8, zipf_s=1.0))
    Eh = np.zeros((N, N), dtype=np.int8)
    Em = np.zeros((N, N), dtype=np.int8)
    Et = np.zeros((N, N), dtype=np.int8)
    rows = []
    # warm-up (imports, caches)
    ghem_ac(ph[0][1], Eh, K, BO, BE, DELTA)
    for m, (kind, D) in enumerate(ph):
        h = ghem_ac(D, Eh, K, BO, BE, DELTA)
        s = solve_exact(D, K, BO, BE, DELTA, E_prev=Em, time_limit=TL)
        t0 = time.perf_counter()
        tm = sad_otr_tm(D, Et, K, BO, BE, DELTA)
        tm_lat = time.perf_counter() - t0
        eps_T = phase_time(D, np.zeros((N, N), dtype=np.int8), 0.0, BE)
        rows.append({
            "family": family, "N": N, "seed": seed, "phase": m, "kind": kind,
            "ghem_T": h["T_penalized"], "milp_T": s["objective"], "milp_status": s["status"],
            "milp_sol_status": s.get("sol_status", ""),
            "gap_pct": max(0.0, 100.0 * (h["T_penalized"] - s["objective"]) / s["objective"]),
            "ghem_ms": 1000 * h["latency_s"], "milp_ms": 1000 * s["wall_s"],
            "tm_T": tm["T"], "tm_ms": 1000 * tm_lat, "tm_slices": tm["n_slices"],
            "lb_T": ideal_lower_bound(D, K, BO, BE), "eps_T": eps_T,
        })
        Eh, Em, Et = h["E"], s["E"], tm["E_last"]
    print(f"done {family} N={N} seed={seed}", flush=True)
    return rows


def main():
    jobs = []
    for family in ("uniform", "skewed"):
        for N in (16, 32, 64, 128, 256):
            for seed in (range(3) if N == 256 else range(5)):
                jobs.append((family, N, seed))
    # largest first for load balance
    jobs.sort(key=lambda j: -j[1])
    with Pool(4) as pool:
        allrows = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    allrows.sort(key=lambda r: (r["family"], r["N"], r["seed"], r["phase"]))
    with (OUT / "scalability_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader(); w.writerows(allrows)
    summ = []
    for family in ("uniform", "skewed"):
        for N in (16, 32, 64, 128, 256):
            sub = [r for r in allrows if r["family"] == family and r["N"] == N]
            if not sub:
                continue
            opt = [r for r in sub if r["milp_status"] == "Optimal" and int(r["milp_sol_status"]) == 1]
            seeds = sorted({r["seed"] for r in sub})
            per_seed_ratio = []
            for sd in seeds:
                ss = [r for r in sub if r["seed"] == sd]
                per_seed_ratio.append((np.mean([r["eps_T"] for r in ss]) / np.mean([r["ghem_T"] for r in ss]),
                                       np.mean([r["eps_T"] for r in ss]) / np.mean([r["tm_T"] for r in ss]),
                                       np.mean([r["tm_T"] for r in ss]) / np.mean([r["lb_T"] for r in ss]),
                                       np.mean([r["ghem_T"] for r in ss]) / np.mean([r["tm_T"] for r in ss])))
            pr = np.array(per_seed_ratio)
            summ.append({
                "family": family, "N": N, "n_seeds": len(seeds), "n_phases": len(sub),
                "n_milp_optimal": len(opt),
                "mean_gap_pct": float(np.mean([r["gap_pct"] for r in opt])) if opt else None,
                "worst_gap_pct": float(np.max([r["gap_pct"] for r in opt])) if opt else None,
                "ghem_mean_ms": float(np.mean([r["ghem_ms"] for r in sub])),
                "ghem_p95_ms": float(np.percentile([r["ghem_ms"] for r in sub], 95)),
                "milp_median_ms": float(np.median([r["milp_ms"] for r in sub])),
                "milp_p95_ms": float(np.percentile([r["milp_ms"] for r in sub], 95)),
                "milp_max_ms": float(np.max([r["milp_ms"] for r in sub])),
                "tm_mean_ms": float(np.mean([r["tm_ms"] for r in sub])),
                "tm_max_ms": float(np.max([r["tm_ms"] for r in sub])),
                "speedup_ghem_vs_eps": float(pr[:, 0].mean()),
                "speedup_tm_vs_eps": float(pr[:, 1].mean()),
                "tm_over_lb": float(pr[:, 2].mean()),
                "ghem_over_tm": float(pr[:, 3].mean()),
            })
    with (OUT / "scalability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    # log-log growth exponents (N >= 32)
    fits = {}
    for family in ("uniform", "skewed"):
        s = [r for r in summ if r["family"] == family and r["N"] >= 32]
        x = np.log([r["N"] for r in s])
        for k in ("ghem_mean_ms", "milp_median_ms", "tm_mean_ms"):
            y = np.log([r[k] for r in s])
            fits[f"{family}_{k}_exponent"] = float(np.polyfit(x, y, 1)[0])
    (OUT / "scalability.json").write_text(json.dumps({"summary": summ, "growth_exponents_N32_256": fits,
                                                      "cbc_time_limit_s": TL, "workers": 4}, indent=2))
    for r in summ:
        print(r)
    print(fits)


if __name__ == "__main__":
    main()
