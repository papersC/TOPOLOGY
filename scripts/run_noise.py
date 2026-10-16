#!/usr/bin/env python3
"""Demand noise / perturbation study (±10–30% volume error)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.closed_form import phase_time
from sad_otr.demands import make_phase_sequence, perturb_demand
from sad_otr.evaluate import FabricParams
from sad_otr.ghem_ac import ghem_ac

OUT = ROOT / "results"


def main() -> None:
    params = FabricParams(seed=0)
    true_phases = make_phase_sequence(params.N, params.n_phases, params.seed)
    noise_levels = [0.0, 0.10, 0.20, 0.30]
    rows = []

    for noise in noise_levels:
        rng = np.random.default_rng(12345 + int(noise * 100))
        E_prev = np.zeros((params.N, params.N), dtype=np.int8)
        Ts_true = []
        Ts_oracle = []
        n_recon = 0
        E_oracle = np.zeros((params.N, params.N), dtype=np.int8)
        for kind, D_true in true_phases:
            D_hat = (
                D_true
                if noise == 0.0
                else perturb_demand(D_true, noise, rng, mode="volume")
            )
            out = ghem_ac(
                D_hat, E_prev, params.K, params.B_opt, params.B_eps, params.delta
            )
            E = out["E"]
            T_comm = phase_time(D_true, E, params.B_opt, params.B_eps)
            T_pen = T_comm + (params.delta if out["reconfigured"] else 0.0)
            Ts_true.append(T_pen)
            if out["reconfigured"]:
                n_recon += 1
            E_prev = E

            out_o = ghem_ac(
                D_true, E_oracle, params.K, params.B_opt, params.B_eps, params.delta
            )
            Ts_oracle.append(out_o["T_penalized"])
            E_oracle = out_o["E"]

        mean_true = float(np.mean(Ts_true))
        mean_oracle = float(np.mean(Ts_oracle))
        degrade = 100.0 * (mean_true - mean_oracle) / mean_oracle if mean_oracle else 0.0
        row = {
            "noise_frac": noise,
            "mean_phase_s_true_eval": round(mean_true, 3),
            "mean_phase_s_oracle": round(mean_oracle, 3),
            "degradation_pct_vs_oracle": round(degrade, 2),
            "n_reconfig": n_recon,
        }
        rows.append(row)
        print(row)

    rng = np.random.default_rng(99)
    E_prev = np.zeros((params.N, params.N), dtype=np.int8)
    Ts = []
    for kind, D_true in true_phases:
        D_hat = (
            perturb_demand(D_true, 0.20, rng, mode="fanout")
            if kind == "EP"
            else D_true
        )
        out = ghem_ac(D_hat, E_prev, params.K, params.B_opt, params.B_eps, params.delta)
        T_comm = phase_time(D_true, out["E"], params.B_opt, params.B_eps)
        T_pen = T_comm + (params.delta if out["reconfigured"] else 0.0)
        Ts.append(T_pen)
        E_prev = out["E"]
    fanout_row = {
        "noise_frac": "fanout_p=0.2_on_EP",
        "mean_phase_s_true_eval": round(float(np.mean(Ts)), 3),
        "mean_phase_s_oracle": rows[0]["mean_phase_s_oracle"],
        "degradation_pct_vs_oracle": round(
            100.0
            * (float(np.mean(Ts)) - rows[0]["mean_phase_s_oracle"])
            / rows[0]["mean_phase_s_oracle"],
            2,
        ),
        "n_reconfig": None,
    }
    rows.append(fanout_row)
    print("fanout:", fanout_row)

    (OUT / "noise.json").write_text(json.dumps({"rows": rows}, indent=2))
    with (OUT / "noise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", OUT / "noise.json")


if __name__ == "__main__":
    main()
