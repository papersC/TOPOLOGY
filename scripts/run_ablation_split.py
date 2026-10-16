#!/usr/bin/env python3
"""Flow-splitting ablation: splittable vs unsplittable vs optical-else-packet."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.closed_form import optical_byte_share, phase_time
from sad_otr.demands import make_phase_sequence
from sad_otr.evaluate import FabricParams, run_scheme
from sad_otr.ghem_ac import ghem_ac
from sad_otr.matching import weight_greedy
from sad_otr.milp import solve_exact

OUT = ROOT / "results"


def optical_else_packet_run(params: FabricParams, phases) -> dict:
    """Once a lightpath exists, all demand on that pair is optical if T allows
    full offload at B_opt; else residual still allowed? Spec: 'optical if
    circuit else full packet' — equivalent to unsplittable assignment.
    Kept as explicit alias measuring the same policy via unsplittable path.
    """
    return run_scheme(params, scheme="ghem", phases=phases, unsplittable=True)


def main() -> None:
    params = FabricParams()
    phases = make_phase_sequence(params.N, params.n_phases, params.seed)

    split = run_scheme(params, scheme="ghem", phases=phases, unsplittable=False)
    unsplit = run_scheme(params, scheme="ghem", phases=phases, unsplittable=True)
    eps = run_scheme(params, scheme="eps", phases=phases)
    static = run_scheme(params, scheme="static_ocs", phases=phases)

    # MILP optima under both policies (for gap)
    def milp_mean(unsplittable: bool) -> tuple[float, float]:
        E_prev = np.zeros((params.N, params.N), dtype=np.int8)
        Ts = []
        opts = []
        tots = []
        for _, D in phases:
            sol = solve_exact(
                D, params.K, params.B_opt, params.B_eps, params.delta,
                E_prev=E_prev, unsplittable=unsplittable, time_limit=180.0,
            )
            Ts.append(sol["objective"])
            opts.append(sol["opt_bytes"])
            tots.append(sol["total_bytes"])
            E_prev = sol["E"]
        mean_T = float(np.mean(Ts))
        opt_pct = 100.0 * sum(opts) / sum(tots) if sum(tots) else 0.0
        return mean_T, opt_pct

    milp_split_T, milp_split_opt = milp_mean(False)
    milp_unsplit_T, milp_unsplit_opt = milp_mean(True)

    def gap(t_h, t_m):
        return max(0.0, 100.0 * (t_h - t_m) / t_m) if t_m > 1e-15 else 0.0

    rows = [
        {
            "policy": "EPS Clos (packet-only)",
            "mean_phase_s": round(eps["mean_phase_s"], 3),
            "optical_pct": round(eps["optical_pct"], 1),
            "gap_vs_milp_pct": None,
            "speedup_vs_eps": 1.0,
        },
        {
            "policy": "Static OCS (splittable eval)",
            "mean_phase_s": round(static["mean_phase_s"], 3),
            "optical_pct": round(static["optical_pct"], 1),
            "gap_vs_milp_pct": None,
            "speedup_vs_eps": round(eps["mean_phase_s"] / static["mean_phase_s"], 2),
        },
        {
            "policy": "GHEM-AC free split",
            "mean_phase_s": round(split["mean_phase_s"], 3),
            "optical_pct": round(split["optical_pct"], 1),
            "gap_vs_milp_pct": round(gap(split["mean_phase_s"], milp_split_T), 2),
            "speedup_vs_eps": round(eps["mean_phase_s"] / split["mean_phase_s"], 2),
        },
        {
            "policy": "GHEM-AC unsplittable",
            "mean_phase_s": round(unsplit["mean_phase_s"], 3),
            "optical_pct": round(unsplit["optical_pct"], 1),
            "gap_vs_milp_pct": round(gap(unsplit["mean_phase_s"], milp_unsplit_T), 2),
            "speedup_vs_eps": round(eps["mean_phase_s"] / unsplit["mean_phase_s"], 2),
        },
        {
            "policy": "MILP free split",
            "mean_phase_s": round(milp_split_T, 3),
            "optical_pct": round(milp_split_opt, 1),
            "gap_vs_milp_pct": 0.0,
            "speedup_vs_eps": round(eps["mean_phase_s"] / milp_split_T, 2),
        },
        {
            "policy": "MILP unsplittable",
            "mean_phase_s": round(milp_unsplit_T, 3),
            "optical_pct": round(milp_unsplit_opt, 1),
            "gap_vs_milp_pct": 0.0,
            "speedup_vs_eps": round(eps["mean_phase_s"] / milp_unsplit_T, 2),
        },
    ]

    # Gain attribution
    # Gain from topology alone (unsplittable) vs additional gain from splitting
    gain_total = eps["mean_phase_s"] - split["mean_phase_s"]
    gain_topo = eps["mean_phase_s"] - unsplit["mean_phase_s"]
    gain_split = unsplit["mean_phase_s"] - split["mean_phase_s"]
    attribution = {
        "eps_mean_s": eps["mean_phase_s"],
        "unsplit_mean_s": unsplit["mean_phase_s"],
        "split_mean_s": split["mean_phase_s"],
        "gain_total_s": gain_total,
        "gain_topology_unsplittable_s": gain_topo,
        "gain_from_splitting_s": gain_split,
        "fraction_of_gain_from_splitting": (gain_split / gain_total) if gain_total > 0 else 0.0,
        "fraction_of_gain_from_topology": (gain_topo / gain_total) if gain_total > 0 else 0.0,
    }
    print("Ablation rows:")
    for r in rows:
        print(r)
    print("Attribution:", attribution)

    payload = {"rows": rows, "attribution": attribution}
    (OUT / "ablation_split.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "ablation_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", OUT / "ablation_split.json")


if __name__ == "__main__":
    main()
