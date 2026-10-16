#!/usr/bin/env python3
"""All schemes on the trace-derived synthetic LLM/MoE workload (final-check revision).

Workload: sad_otr.llm_workload (DeepSeek-V3 published dimensions, assumed
4-stage x 8-rank deployment on N=32 racks).  Fabric: K=4, B_opt=800 Gbps,
B_eps=400 Gbps.  delta=15 ms on seeds 0-9; delta in {1, 100} ms on seeds 0-4.
Outputs: results/llm_seeds.csv, results/llm_summary.json, results/llm_phasekind.csv
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
sys.path.insert(0, str(ROOT / "scripts"))
from run_baselines import SCHEMES, run_all, ci95  # noqa: E402
from sad_otr.evaluate import FabricParams  # noqa: E402
from sad_otr.llm_workload import LLM_CONFIG, make_llm_phase_sequence  # noqa: E402

OUT = ROOT / "results"


def main() -> None:
    rows, kind_rows = [], []
    for delta, seeds in ((0.015, range(10)), (0.001, range(5)), (0.100, range(5))):
        for seed in seeds:
            p = FabricParams(seed=seed, delta=delta)
            phases = make_llm_phase_sequence(seed)
            res = run_all(p, phases)
            eps = res["EPS Clos"]["mean_phase_s"]
            for name in SCHEMES:
                r = res[name]
                rows.append({"delta_ms": delta * 1000, "seed": seed, "scheme": name,
                             "mean_phase_s": r["mean_phase_s"], "max_phase_s": r["max_phase_s"],
                             "optical_pct": r["optical_pct"], "n_reconfig": r["n_reconfig"],
                             "speedup_vs_eps": eps / r["mean_phase_s"],
                             "mean_latency_ms": r["mean_latency_ms"]})
                if delta == 0.015:
                    for kd in ("EPD", "EPC", "PP", "DP"):
                        vals = [t for t, k in zip(r["per_phase"], r["kinds"]) if k == kd]
                        kind_rows.append({"seed": seed, "scheme": name, "kind": kd,
                                          "mean_s": float(np.mean(vals))})
            print(f"delta {delta*1000:g} ms seed {seed}: " +
                  ", ".join(f"{n}={res[n]['mean_phase_s']:.4f}" for n in SCHEMES), flush=True)
    with (OUT / "llm_seeds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / "llm_phasekind.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kind_rows[0])); w.writeheader(); w.writerows(kind_rows)
    summ = {}
    for dms in (15.0, 1.0, 100.0):
        summ[str(dms)] = {}
        tm = {r["seed"]: r["mean_phase_s"] for r in rows if r["delta_ms"] == dms and r["scheme"] == "SAD-OTR-TM"}
        gh = {r["seed"]: r["mean_phase_s"] for r in rows if r["delta_ms"] == dms and r["scheme"] == "SAD-OTR (GHEM-AC)"}
        for name in SCHEMES:
            sub = [r for r in rows if r["delta_ms"] == dms and r["scheme"] == name]
            d = {k: dict(zip(("mean", "std", "ci95_half"), ci95([r[k] for r in sub])))
                 for k in ("mean_phase_s", "max_phase_s", "optical_pct", "n_reconfig", "speedup_vs_eps", "mean_latency_ms")}
            d["ratio_to_tm"] = dict(zip(("mean", "std", "ci95_half"), ci95([r["mean_phase_s"] / tm[r["seed"]] for r in sub])))
            d["ratio_to_ghem"] = dict(zip(("mean", "std", "ci95_half"), ci95([r["mean_phase_s"] / gh[r["seed"]] for r in sub])))
            summ[str(dms)][name] = d
    # paired tests at 15 ms
    tests = {}
    by = {}
    for r in rows:
        if r["delta_ms"] == 15.0:
            by.setdefault(r["scheme"], {})[r["seed"]] = r["mean_phase_s"]
    for a, b in (("SAD-OTR-TM", "Eclipse-K+LP"), ("SAD-OTR-TM", "SAD-OTR (GHEM-AC)"),
                 ("SAD-OTR (GHEM-AC)", "MWM per phase"), ("SAD-OTR-TM", "Eclipse-K")):
        x = np.array([by[a][s] for s in range(10)]); y = np.array([by[b][s] for s in range(10)])
        t = stats.ttest_rel(np.log(x), np.log(y))
        diff = x - y
        if np.all(np.abs(diff) < 1e-12):
            pw = 1.0
        else:
            pw = float(stats.wilcoxon(x, y, method="exact").pvalue)
        tests[f"{a} vs {b}"] = {"mean_ratio": float((x / y).mean()), "wins_a": int((x < y).sum()),
                                "ties": int((np.abs(diff) < 1e-12).sum()),
                                "p_t_log": float(t.pvalue), "p_wilcoxon": pw}
    (OUT / "llm_summary.json").write_text(json.dumps({"config": LLM_CONFIG, "summary": summ, "tests": tests}, indent=2))
    for dms, s in summ.items():
        print("delta", dms)
        for name, d in s.items():
            print(f"  {name:20s} {d['mean_phase_s']['mean']:.4f}±{d['mean_phase_s']['ci95_half']:.4f} "
                  f"x{d['speedup_vs_eps']['mean']:.2f} opt {d['optical_pct']['mean']:.1f}% rec {d['n_reconfig']['mean']:.1f} "
                  f"lat {d['mean_latency_ms']['mean']:.1f} /TM {d['ratio_to_tm']['mean']:.3f}")
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    main()
