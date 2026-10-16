#!/usr/bin/env python3
"""Demand-uncertainty study over seeds 0-9 (Q1 revision; extends run_noise.py).

The controller decides on a perturbed matrix D_hat; completion time is
evaluated on the true matrix D.
  * SAD-OTR (GHEM-AC): topology from D_hat, T = T(D, E) + delta if changed.
  * SAD-OTR-TM: slice sequence and durations planned on D_hat, then executed
    on D with fixed durations (circuits carry what they can, the packet layer
    the rest; baselines.run_schedule accounting).
Degradation is relative to the same policy with exact knowledge (oracle).
Outputs: results/robust.csv, results/robust.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.baselines import run_schedule  # noqa: E402
from sad_otr.closed_form import phase_time  # noqa: E402
from sad_otr.demands import make_phase_sequence, perturb_demand  # noqa: E402
from sad_otr.ghem_ac import ghem_ac  # noqa: E402
from sad_otr.multislot import sad_otr_tm  # noqa: E402

OUT = ROOT / "results"
N, K, BO, BE, DELTA = 32, 4, 800.0, 400.0, 0.015


def run_policy(phases, hats):
    Eg = np.zeros((N, N), dtype=np.int8)
    Et = np.zeros((N, N), dtype=np.int8)
    Tg, Tt = [], []
    for (kind, D), Dh in zip(phases, hats):
        g = ghem_ac(Dh, Eg, K, BO, BE, DELTA)
        Tg.append(phase_time(D, g["E"], BO, BE) + (DELTA if g["reconfigured"] else 0.0))
        Eg = g["E"]
        t = sad_otr_tm(Dh, Et, K, BO, BE, DELTA)
        sched = list(zip(t["configs"], t["durations"]))
        r = run_schedule(D, sched, BO, BE, DELTA, first_free=(t["charges"][0] == 0.0))
        Tt.append(r["T"])
        Et = t["E_last"]
    return float(np.mean(Tg)), float(np.mean(Tt))


def ci(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    seeds = range(10)
    settings = [("volume", f) for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)] + \
               [("fanout", p) for p in (0.1, 0.2, 0.3, 0.5)]
    oracle = {}
    for sd in seeds:
        ph = make_phase_sequence(N, 12, sd)
        oracle[sd] = run_policy(ph, [D for _, D in ph])
    rows = []
    for mode, lvl in settings:
        dg, dt, ag, at = [], [], [], []
        for sd in seeds:
            ph = make_phase_sequence(N, 12, sd)
            rng = np.random.default_rng(1000 * sd + int(lvl * 100) + (0 if mode == "volume" else 500))
            hats = []
            for kind, D in ph:
                if lvl == 0.0:
                    hats.append(D)
                elif mode == "volume":
                    hats.append(perturb_demand(D, lvl, rng, mode="volume"))
                else:
                    hats.append(perturb_demand(D, lvl, rng, mode="fanout") if kind == "EP" else D)
            g, t = run_policy(ph, hats)
            ag.append(g); at.append(t)
            dg.append(100 * (g - oracle[sd][0]) / oracle[sd][0])
            dt.append(100 * (t - oracle[sd][1]) / oracle[sd][1])
        mg, hg = ci(dg)
        mt, ht = ci(dt)
        ga, gh = ci(ag)
        ta, th = ci(at)
        rows.append({"mode": mode, "level": lvl, "ghem_degr_pct": mg, "ghem_ci95": hg,
                     "tm_degr_pct": mt, "tm_ci95": ht,
                     "ghem_abs_s": ga, "ghem_abs_ci95": gh, "tm_abs_s": ta, "tm_abs_ci95": th})
        print(rows[-1], flush=True)
    with (OUT / "robust.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / "robust.json").write_text(json.dumps({"rows": rows, "oracle_mean_s": oracle}, indent=2))


if __name__ == "__main__":
    main()
