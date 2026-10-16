#!/usr/bin/env python3
"""Generate core Tables I–IV style results (seed=0) with destination EPS capacity."""

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
OUT.mkdir(parents=True, exist_ok=True)


def speedup(base: float, t: float) -> float:
    return base / t if t > 1e-15 else float("inf")


def main() -> None:
    params = FabricParams()
    phases = make_phase_sequence(params.N, params.n_phases, params.seed)

    # --- Table agg ---
    schemes = ["eps", "static_ocs", "ghem", "ghem_delta0"]
    agg_rows = []
    results = {}
    for s in schemes:
        r = run_scheme(params, scheme=s, phases=phases, use_repair=False)
        results[s] = r
        agg_rows.append(r)
        print(
            f"{s}: mean={r['mean_phase_s']:.4f}s max={r['max_phase_s']:.4f}s "
            f"opt%={r['optical_pct']:.1f} recon={r['n_reconfig']}"
        )

    eps_mean = results["eps"]["mean_phase_s"]
    table_agg = []
    labels = {
        "eps": "EPS Clos",
        "static_ocs": "Static OCS",
        "ghem": "SAD-OTR (GHEM-AC)",
        "ghem_delta0": "SAD-OTR, delta=0",
    }
    for s in schemes:
        r = results[s]
        table_agg.append({
            "scheme": labels[s],
            "mean_phase_s": round(r["mean_phase_s"], 3),
            "max_phase_s": round(r["max_phase_s"], 3),
            "optical_pct": round(r["optical_pct"], 1),
            "n_reconfig": r["n_reconfig"],
            "speedup_vs_eps": round(speedup(eps_mean, r["mean_phase_s"]), 2),
        })

    # With repair variant for diagnosis
    r_repair = run_scheme(params, scheme="ghem", phases=phases, use_repair=True)
    results["ghem_repair"] = r_repair
    print(
        f"ghem+repair: mean={r_repair['mean_phase_s']:.4f}s "
        f"opt%={r_repair['optical_pct']:.1f} recon={r_repair['n_reconfig']}"
    )

    # --- Optimality gaps vs MILP across N ---
    gap_rows = []
    for N in [8, 12, 16, 20, 24, 28, 32]:
        p = FabricParams(N=N)
        ph = make_phase_sequence(N, p.n_phases, p.seed)
        E_prev_h = np.zeros((N, N), dtype=np.int8)
        E_prev_m = np.zeros((N, N), dtype=np.int8)
        gaps = []
        ghem_lats = []
        milp_lats = []
        worst = {"gap": -1.0, "phase": -1, "kind": "", "winner": "", "scores": {}}
        for m, (kind, D) in enumerate(ph):
            hout = ghem_ac(D, E_prev_h, p.K, p.B_opt, p.B_eps, p.delta)
            mout = solve_exact(
                D, p.K, p.B_opt, p.B_eps, p.delta, E_prev=E_prev_m, time_limit=180.0
            )
            # Gap on penalized objective
            t_h = hout["T_penalized"]
            t_m = mout["objective"]
            gap = 100.0 * (t_h - t_m) / t_m if t_m > 1e-15 else 0.0
            # Numerical noise: clamp tiny negatives
            gap = max(0.0, gap)
            gaps.append(gap)
            ghem_lats.append(hout["latency_s"] * 1000.0)
            milp_lats.append(mout["wall_s"] * 1000.0)
            if gap > worst["gap"]:
                worst = {
                    "gap": gap,
                    "phase": m,
                    "kind": kind,
                    "winner": hout["winner"],
                    "scores": hout["candidate_scores"],
                    "T_h": t_h,
                    "T_m": t_m,
                }
            E_prev_h = hout["E"]
            E_prev_m = mout["E"]

        # Repair pass for same instances (gap mitigation report)
        E_prev_r = np.zeros((N, N), dtype=np.int8)
        E_prev_m2 = np.zeros((N, N), dtype=np.int8)
        gaps_r = []
        for m, (kind, D) in enumerate(ph):
            hout = ghem_ac(
                D, E_prev_r, p.K, p.B_opt, p.B_eps, p.delta, use_repair=True
            )
            mout = solve_exact(
                D, p.K, p.B_opt, p.B_eps, p.delta, E_prev=E_prev_m2, time_limit=180.0
            )
            gap = max(0.0, 100.0 * (hout["T_penalized"] - mout["objective"]) / mout["objective"])
            gaps_r.append(gap)
            E_prev_r = hout["E"]
            E_prev_m2 = mout["E"]

        row = {
            "N": N,
            "mean_gap_pct": round(float(np.mean(gaps)), 2),
            "worst_gap_pct": round(float(np.max(gaps)), 2),
            "mean_gap_repair_pct": round(float(np.mean(gaps_r)), 2),
            "worst_gap_repair_pct": round(float(np.max(gaps_r)), 2),
            "ghem_ms": round(float(np.mean(ghem_lats)), 1),
            "milp_ms": round(float(np.mean(milp_lats)), 1),
            "worst_phase": worst["phase"],
            "worst_kind": worst["kind"],
            "worst_winner": worst["winner"],
            "worst_scores": worst["scores"],
        }
        gap_rows.append(row)
        print(
            f"N={N}: mean_gap={row['mean_gap_pct']}% worst={row['worst_gap_pct']}% "
            f"repair_worst={row['worst_gap_repair_pct']}% "
            f"ghem={row['ghem_ms']}ms milp={row['milp_ms']}ms "
            f"worst@{worst['phase']}:{worst['kind']} winner={worst['winner']}"
        )

    # --- K sweep ---
    k_rows = []
    for K in [1, 2, 4, 6, 8, 12, 16]:
        p = FabricParams(K=K)
        r = run_scheme(p, scheme="ghem", phases=make_phase_sequence(p.N, p.n_phases, p.seed))
        e = run_scheme(p, scheme="eps", phases=make_phase_sequence(p.N, p.n_phases, p.seed))
        k_rows.append({
            "K": K,
            "mean_phase_s": round(r["mean_phase_s"], 3),
            "speedup": round(speedup(e["mean_phase_s"], r["mean_phase_s"]), 2),
            "optical_pct": round(r["optical_pct"], 1),
        })
        print(f"K={K}: {k_rows[-1]}")

    # --- B_eps sweep ---
    beps_rows = []
    for B_eps in [100, 200, 400, 800, 1600]:
        p = FabricParams(B_eps=float(B_eps))
        ph = make_phase_sequence(p.N, p.n_phases, p.seed)
        e = run_scheme(p, scheme="eps", phases=ph)
        s = run_scheme(p, scheme="static_ocs", phases=ph)
        g = run_scheme(p, scheme="ghem", phases=ph)
        beps_rows.append({
            "B_eps": B_eps,
            "eps_s": round(e["mean_phase_s"], 3),
            "static_s": round(s["mean_phase_s"], 3),
            "sad_s": round(g["mean_phase_s"], 3),
            "speedup": round(speedup(e["mean_phase_s"], g["mean_phase_s"]), 2),
        })
        print(f"B_eps={B_eps}: {beps_rows[-1]}")

    payload = {
        "table_agg": table_agg,
        "table_opt": gap_rows,
        "table_k": k_rows,
        "table_beps": beps_rows,
        "ghem_repair_agg": {
            "mean_phase_s": round(r_repair["mean_phase_s"], 3),
            "max_phase_s": round(r_repair["max_phase_s"], 3),
            "optical_pct": round(r_repair["optical_pct"], 1),
            "n_reconfig": r_repair["n_reconfig"],
            "speedup_vs_eps": round(speedup(eps_mean, r_repair["mean_phase_s"]), 2),
        },
        "notes": {
            "seed": 0,
            "destination_eps_capacity": True,
            "static_ocs": "WeightGreedy on phase-1, held fixed",
            "bisection_eps_s": 1e-9,
            "cbc": "PULP_CBC_CMD timeLimit=180 gapRel=0 threads=1; wall includes model build",
        },
    }

    (OUT / "main_tables.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "table_agg.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_agg[0].keys()))
        w.writeheader()
        w.writerows(table_agg)
    with (OUT / "table_opt.csv").open("w", newline="") as f:
        # flatten without nested scores for CSV
        fields = [
            "N", "mean_gap_pct", "worst_gap_pct", "mean_gap_repair_pct",
            "worst_gap_repair_pct", "ghem_ms", "milp_ms",
            "worst_phase", "worst_kind", "worst_winner",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in gap_rows:
            w.writerow({k: row[k] for k in fields})
    with (OUT / "table_k.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(k_rows[0].keys()))
        w.writeheader()
        w.writerows(k_rows)
    with (OUT / "table_beps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(beps_rows[0].keys()))
        w.writeheader()
        w.writerows(beps_rows)

    print("Wrote", OUT / "main_tables.json")


if __name__ == "__main__":
    main()
