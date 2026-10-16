#!/usr/bin/env python3
"""Multi-seed statistics (>=10 seeds) for Table I metrics and gaps."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.demands import make_phase_sequence
from sad_otr.evaluate import FabricParams, run_scheme
from sad_otr.ghem_ac import ghem_ac
from sad_otr.milp import solve_exact

OUT = ROOT / "results"


def main() -> None:
    seeds = list(range(10))
    per_seed = []

    for seed in seeds:
        p = FabricParams(seed=seed)
        phases = make_phase_sequence(p.N, p.n_phases, seed)
        eps = run_scheme(p, scheme="eps", phases=phases)
        static = run_scheme(p, scheme="static_ocs", phases=phases)
        ghem = run_scheme(p, scheme="ghem", phases=phases)

        E_h = np.zeros((p.N, p.N), dtype=np.int8)
        E_m = np.zeros((p.N, p.N), dtype=np.int8)
        gaps = []
        for _, D in phases:
            h = ghem_ac(D, E_h, p.K, p.B_opt, p.B_eps, p.delta)
            m = solve_exact(
                D, p.K, p.B_opt, p.B_eps, p.delta, E_prev=E_m, time_limit=120.0
            )
            gap = max(0.0, 100.0 * (h["T_penalized"] - m["objective"]) / m["objective"])
            gaps.append(gap)
            E_h = h["E"]
            E_m = m["E"]

        row = {
            "seed": seed,
            "eps_mean_s": eps["mean_phase_s"],
            "static_mean_s": static["mean_phase_s"],
            "ghem_mean_s": ghem["mean_phase_s"],
            "ghem_max_s": ghem["max_phase_s"],
            "optical_pct": ghem["optical_pct"],
            "n_reconfig": ghem["n_reconfig"],
            "speedup_vs_eps": eps["mean_phase_s"] / ghem["mean_phase_s"],
            "speedup_vs_static": static["mean_phase_s"] / ghem["mean_phase_s"],
            "mean_gap_pct": float(np.mean(gaps)),
            "worst_gap_pct": float(np.max(gaps)),
        }
        per_seed.append(row)
        print(
            f"seed={seed}: ghem={row['ghem_mean_s']:.3f}s "
            f"opt%={row['optical_pct']:.1f} "
            f"gap={row['mean_gap_pct']:.2f}/{row['worst_gap_pct']:.2f}"
        )

    keys = [k for k in per_seed[0] if k != "seed"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in per_seed], dtype=float)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    payload = {"seeds": seeds, "per_seed": per_seed, "summary": summary}
    (OUT / "multiseed.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "multiseed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)
    print(
        "Summary mean±std ghem_mean_s:",
        f"{summary['ghem_mean_s']['mean']:.3f}±{summary['ghem_mean_s']['std']:.3f}",
    )
    print("Wrote", OUT / "multiseed.json")


if __name__ == "__main__":
    main()
