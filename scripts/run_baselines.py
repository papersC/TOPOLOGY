#!/usr/bin/env python3
"""Literature baselines, ideal bound, and slice-level SAD-OTR-TM (Q1 revision).

Outputs
  results/baselines_seeds.csv     per-seed, per-scheme aggregates (seeds 0-9)
  results/baselines_summary.json  mean, std, 95% t-CI per scheme and metric
  results/baselines_phasekind.csv mean phase time by phase kind (seed 0)
  results/baselines_delta.csv     delta sweep, seeds 0-4 (mean and 95% CI)
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

from sad_otr.baselines import (eclipse_k, ideal_lower_bound, mwm_topology,  # noqa: E402
                               rotor_time, solstice_k)
from sad_otr.closed_form import optical_byte_share, phase_time  # noqa: E402
from sad_otr.demands import make_phase_sequence  # noqa: E402
from sad_otr.evaluate import FabricParams, run_scheme  # noqa: E402
from sad_otr.multislot import best_retimed_prefix, sad_otr_tm  # noqa: E402

OUT = ROOT / "results"

SCHEMES = [
    "EPS Clos", "Static OCS", "RotorNet-style", "MWM per phase",
    "Solstice-K", "Solstice-K+LP", "Eclipse-K", "Eclipse-K+LP",
    "SAD-OTR (GHEM-AC)", "SAD-OTR-TM", "Ideal bound",
]


def run_all(p: FabricParams, phases, schemes=SCHEMES) -> dict[str, dict]:
    N, K, Bo, Be, d = p.N, p.K, p.B_opt, p.B_eps, p.delta
    res: dict[str, dict] = {}

    def agg(name, Ts, ob, tb, nrec, lat, kinds):
        res[name] = {
            "mean_phase_s": float(np.mean(Ts)), "max_phase_s": float(np.max(Ts)),
            "optical_pct": 100.0 * ob / tb if tb > 0 else 0.0,
            "n_reconfig": int(nrec), "mean_latency_ms": 1000.0 * float(np.mean(lat)),
            "per_phase": [float(t) for t in Ts], "kinds": kinds,
        }

    kinds = [k for k, _ in phases]
    if "EPS Clos" in schemes:
        r = run_scheme(p, "eps", phases=phases)
        agg("EPS Clos", [x["T"] for x in r["phases"]], 0.0, sum(x["total_bytes"] for x in r["phases"]), 0, [0], kinds)
    if "Static OCS" in schemes:
        r = run_scheme(p, "static_ocs", phases=phases)
        agg("Static OCS", [x["T"] for x in r["phases"]], sum(x["opt_bytes"] for x in r["phases"]),
            sum(x["total_bytes"] for x in r["phases"]), 0, [0], kinds)
    if "SAD-OTR (GHEM-AC)" in schemes:
        r = run_scheme(p, "ghem", phases=phases)
        agg("SAD-OTR (GHEM-AC)", [x["T"] for x in r["phases"]], sum(x["opt_bytes"] for x in r["phases"]),
            sum(x["total_bytes"] for x in r["phases"]), r["n_reconfig"], [x["latency_s"] for x in r["phases"]], kinds)

    if "RotorNet-style" in schemes:
        Ts, ob, tb = [], 0.0, 0.0
        for _, D in phases:
            o = rotor_time(D, K, Bo, Be)
            Ts.append(o["T"]); ob += o["opt_bytes"]; tb += o["total_bytes"]
        agg("RotorNet-style", Ts, ob, tb, 0, [0], kinds)

    if "MWM per phase" in schemes:
        Ts, ob, tb, nrec, lat = [], 0.0, 0.0, 0, []
        E_prev = np.zeros((N, N), dtype=np.int8)
        for _, D in phases:
            t0 = time.perf_counter()
            E = mwm_topology(D, K)
            lat.append(time.perf_counter() - t0)
            ch = d if np.any(E != E_prev) else 0.0
            nrec += ch > 0
            Tc = phase_time(D, E, Bo, Be)
            o, t = optical_byte_share(D, E, Bo, Tc)
            Ts.append(Tc + ch); ob += o; tb += t
            E_prev = E
        agg("MWM per phase", Ts, ob, tb, nrec, lat, kinds)

    for base, fn in (("Solstice-K", solstice_k), ("Eclipse-K", eclipse_k)):
        if base not in schemes and base + "+LP" not in schemes:
            continue
        Ts, ob, tb, nrec, lat = [], 0.0, 0.0, 0, []
        Ts2, ob2, nrec2, lat2 = [], 0.0, 0, []
        for _, D in phases:
            t0 = time.perf_counter()
            o = fn(D, K, Bo, Be, d)
            t1 = time.perf_counter()
            Ts.append(o["T"]); ob += o["opt_bytes"]; tb += o["total_bytes"]; nrec += o["n_configs"]
            lat.append(t1 - t0)
            cfgs = [E for E, _ in o["schedule"]]
            if cfgs:
                rr = best_retimed_prefix(D, cfgs, [d] * len(cfgs), Bo, Be, min_len=0)
            else:
                rr = {"T": o["T"], "opt_bytes": 0.0, "len": 0}
            lat2.append(time.perf_counter() - t0)
            Ts2.append(rr["T"]); ob2 += rr["opt_bytes"]; nrec2 += rr["len"]
        agg(base, Ts, ob, tb, nrec, lat, kinds)
        agg(base + "+LP", Ts2, ob2, tb, nrec2, lat2, kinds)

    if "SAD-OTR-TM" in schemes:
        Ts, ob, tb, nrec, lat = [], 0.0, 0.0, 0, []
        E_prev = np.zeros((N, N), dtype=np.int8)
        for _, D in phases:
            t0 = time.perf_counter()
            o = sad_otr_tm(D, E_prev, K, Bo, Be, d)
            lat.append(time.perf_counter() - t0)
            Ts.append(o["T"]); ob += o["opt_bytes"]; tb += o["total_bytes"]; nrec += o["n_charged"]
            E_prev = o["E_last"]
        agg("SAD-OTR-TM", Ts, ob, tb, nrec, lat, kinds)

    if "Ideal bound" in schemes:
        Ts = [ideal_lower_bound(D, K, Bo, Be) for _, D in phases]
        agg("Ideal bound", Ts, float("nan"), 1.0, 0, [0], kinds)
    return res


def ci95(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, 0.0, 0.0
    s = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n))
    return m, s, h


def main() -> None:
    seeds = list(range(10))
    rows = []
    per_seed = {}
    for seed in seeds:
        p = FabricParams(seed=seed)
        phases = make_phase_sequence(p.N, p.n_phases, seed)
        res = run_all(p, phases)
        per_seed[seed] = res
        eps = res["EPS Clos"]["mean_phase_s"]
        for name in SCHEMES:
            r = res[name]
            rows.append({
                "seed": seed, "scheme": name,
                "mean_phase_s": r["mean_phase_s"], "max_phase_s": r["max_phase_s"],
                "optical_pct": r["optical_pct"], "n_reconfig": r["n_reconfig"],
                "speedup_vs_eps": eps / r["mean_phase_s"],
                "mean_latency_ms": r["mean_latency_ms"],
            })
        print(f"seed {seed}: " + ", ".join(f"{n}={res[n]['mean_phase_s']:.3f}" for n in SCHEMES), flush=True)

    with (OUT / "baselines_seeds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {}
    for name in SCHEMES:
        sub = [r for r in rows if r["scheme"] == name]
        summary[name] = {}
        for k in ("mean_phase_s", "max_phase_s", "optical_pct", "n_reconfig", "speedup_vs_eps", "mean_latency_ms"):
            m, s, h = ci95([r[k] for r in sub])
            summary[name][k] = {"mean": m, "std": s, "ci95_half": h}
        # seed-0 exact values
        s0 = [r for r in sub if r["seed"] == 0][0]
        summary[name]["seed0"] = {k: s0[k] for k in s0 if k not in ("seed", "scheme")}
    # paired comparisons vs SAD-OTR (GHEM-AC)
    paired = {}
    g = np.array([r["mean_phase_s"] for r in rows if r["scheme"] == "SAD-OTR (GHEM-AC)"])
    tm = np.array([r["mean_phase_s"] for r in rows if r["scheme"] == "SAD-OTR-TM"])
    for name in SCHEMES:
        x = np.array([r["mean_phase_s"] for r in rows if r["scheme"] == name])
        m, s, h = ci95(x / g)
        m2, s2, h2 = ci95(x / tm)
        paired[name] = {"ratio_to_ghem": {"mean": m, "ci95_half": h},
                        "ratio_to_tm": {"mean": m2, "ci95_half": h2}}
    # phase-kind breakdown, seed 0
    kind_rows = []
    res0 = per_seed[0]
    for name in SCHEMES:
        r = res0[name]
        row = {"scheme": name}
        for kd in ("TP", "EP", "PP"):
            vals = [t for t, k in zip(r["per_phase"], r["kinds"]) if k == kd]
            row[f"{kd}_mean_s"] = float(np.mean(vals))
        kind_rows.append(row)
    with (OUT / "baselines_phasekind.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kind_rows[0].keys()))
        w.writeheader(); w.writerows(kind_rows)
    # per-phase seed 0
    with (OUT / "baselines_phases_seed0.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "kind"] + SCHEMES)
        for m in range(len(res0["EPS Clos"]["per_phase"])):
            w.writerow([m, res0["EPS Clos"]["kinds"][m]] + [res0[n]["per_phase"][m] for n in SCHEMES])

    # delta sweep, seeds 0-4
    deltas_ms = [0.1, 1.0, 5.0, 15.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    dschemes = ["EPS Clos", "Static OCS", "MWM per phase", "Solstice-K+LP", "Eclipse-K", "Eclipse-K+LP",
                "SAD-OTR (GHEM-AC)", "SAD-OTR-TM", "Ideal bound"]
    drows = []
    for dm in deltas_ms:
        acc = {n: [] for n in dschemes}
        rec = {n: [] for n in dschemes}
        for seed in range(5):
            p = FabricParams(seed=seed, delta=dm / 1000.0)
            phases = make_phase_sequence(p.N, p.n_phases, seed)
            res = run_all(p, phases, schemes=dschemes + (["Solstice-K"] if "Solstice-K+LP" in dschemes else []))
            for n in dschemes:
                acc[n].append(res[n]["mean_phase_s"]); rec[n].append(res[n]["n_reconfig"])
        for n in dschemes:
            m, s, h = ci95(acc[n])
            drows.append({"delta_ms": dm, "scheme": n, "mean_phase_s": m, "ci95_half": h,
                          "mean_reconfigs": float(np.mean(rec[n]))})
        print(f"delta {dm} ms: " + ", ".join(f"{n}={np.mean(acc[n]):.3f}" for n in dschemes), flush=True)
    with (OUT / "baselines_delta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
        w.writeheader(); w.writerows(drows)

    payload = {
        "seeds": seeds,
        "params": {"N": 32, "K": 4, "B_opt_gbps": 800, "B_eps_gbps": 400, "delta_s": 0.015, "n_phases": 12},
        "notes": {
            "convention": "packet layer counted only during transmission; T = transmission + delta per charged reconfiguration (all schemes)",
            "rotor_duty_cycle": 0.9,
            "solstice": "QuickStuff + BigSlice; up to K disjoint perfect matchings per slice; Solstice termination",
            "eclipse": "greedy argmax gain/(alpha+delta); K rounds of Hungarian; best prefix for min completion",
            "+LP": "configuration sequence kept; durations and optical/packet split re-optimised by LP; best prefix",
            "ci": "Student-t 95% CI over seeds",
        },
        "summary": summary, "paired": paired,
    }
    (OUT / "baselines_summary.json").write_text(json.dumps(payload, indent=2))
    print("Wrote baselines_*")


if __name__ == "__main__":
    main()
