#!/usr/bin/env python3
"""Decision-latency timing harness."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.demands import make_phase_sequence
from sad_otr.evaluate import FabricParams
from sad_otr.ghem_ac import ghem_ac
from sad_otr.milp import solve_exact

OUT = ROOT / "results"


def median_ms(samples_s: list[float]) -> float:
    return float(np.median(samples_s) * 1000.0)


def main() -> None:
    sizes = [8, 12, 16, 20, 24, 28, 32, 64, 128]
    reps_ghem = 21
    reps_milp = 5
    rows = []

    for N in sizes:
        p = FabricParams(N=N)
        phases = make_phase_sequence(N, p.n_phases, seed=0)
        kind, D = phases[1] if len(phases) > 1 else phases[0]
        E0 = np.zeros((N, N), dtype=np.int8)

        ghem_ac(D, E0, p.K, p.B_opt, p.B_eps, p.delta)
        g_samples = []
        for _ in range(reps_ghem - 1):
            out = ghem_ac(D, E0, p.K, p.B_opt, p.B_eps, p.delta)
            g_samples.append(out["latency_s"])

        solve_exact(D, p.K, p.B_opt, p.B_eps, p.delta, E_prev=E0, time_limit=120.0)
        m_wall, m_solve, m_build = [], [], []
        for _ in range(max(1, reps_milp - 1)):
            sol = solve_exact(
                D, p.K, p.B_opt, p.B_eps, p.delta, E_prev=E0, time_limit=120.0
            )
            m_wall.append(sol["wall_s"])
            m_solve.append(sol["solve_s"])
            m_build.append(sol["build_s"])

        E_h = np.zeros((N, N), dtype=np.int8)
        E_m = np.zeros((N, N), dtype=np.int8)
        suite_g, suite_m = [], []
        for _, Dd in phases:
            hout = ghem_ac(Dd, E_h, p.K, p.B_opt, p.B_eps, p.delta)
            mout = solve_exact(
                Dd, p.K, p.B_opt, p.B_eps, p.delta, E_prev=E_m, time_limit=120.0
            )
            suite_g.append(hout["latency_s"])
            suite_m.append(mout["wall_s"])
            E_h = hout["E"]
            E_m = mout["E"]

        row = {
            "N": N,
            "ghem_median_ms_single": round(median_ms(g_samples), 1),
            "ghem_mean_ms_suite": round(float(np.mean(suite_g) * 1000), 1),
            "milp_median_wall_ms_single": round(median_ms(m_wall), 1),
            "milp_mean_wall_ms_suite": round(float(np.mean(suite_m) * 1000), 1),
            "milp_median_solve_ms": round(median_ms(m_solve), 1),
            "milp_median_build_ms": round(median_ms(m_build), 1),
            "phase_kind_timed": kind,
        }
        rows.append(row)
        print(row)

    methodology = {
        "language": "Python 3 (CPython), unoptimized reference implementation",
        "timer": "time.perf_counter",
        "ghem": "warm start discarded; median of remaining single-phase reps; Table II uses mean over 12-phase suite wall times",
        "milp": "PuLP model build + CBC solve included in wall; PULP_CBC_CMD timeLimit=120s gapRel=0 threads=1 msg=False",
        "bisection_eps_s": 1e-9,
        "destination_eps_capacity": True,
    }
    payload = {"rows": rows, "methodology": methodology}
    (OUT / "latency.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "latency.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", OUT / "latency.json")


if __name__ == "__main__":
    main()
