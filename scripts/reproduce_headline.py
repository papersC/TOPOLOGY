#!/usr/bin/env python3
"""Reproduce the headline comparison (Table baselines / tab:sig inputs) from
exported input files and check it against results/baselines_seeds.csv.

Usage: python scripts/reproduce_headline.py --inputs ../inputs [--seeds 0 1 ...] [--llm]

For every seed it loads inputs/primary/primary_N32_seed<s>.npz, verifies that
the file equals the on-the-fly generator output, runs all eleven schemes
(delta = 15 ms, N=32, K=4) and compares the mean phase time of every scheme
with the stored result (tolerance 1e-6 relative; decision latencies are
wall-clock and are not compared).  With --llm it does the same for the
DeepSeek-V3-derived workload (results/llm_seeds.csv, delta = 15 ms).
Writes results_reproduced/headline_check.json.  Exit code 0 iff all match.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_baselines import SCHEMES, run_all  # noqa: E402
from sad_otr.demands import make_phase_sequence  # noqa: E402
from sad_otr.evaluate import FabricParams  # noqa: E402
from sad_otr.inputs_io import load_phases  # noqa: E402
from sad_otr.llm_workload import make_llm_phase_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(ROOT.parent / "inputs"))
    ap.add_argument("--results", default=None, help="results dir (default: ../results or ./results)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    inp = Path(a.inputs)
    res = Path(a.results) if a.results else (ROOT.parent / "results" if (ROOT.parent / "results").exists() else ROOT / "results")
    jobs = [("primary", "baselines_seeds.csv", lambda s: inp / f"primary/primary_N32_seed{s}.npz",
             lambda s: make_phase_sequence(32, 12, s), None)]
    if a.llm:
        jobs.append(("llm", "llm_seeds.csv", lambda s: inp / f"llm/llm_seed{s}.npz", make_llm_phase_sequence, "15.0"))
    p = FabricParams(N=32, K=4, B_opt=800.0, B_eps=400.0, delta=0.015)
    report, bad = {}, 0
    for name, fcsv, fpath, gen, dfilt in jobs:
        ref = {}
        for r in csv.DictReader(open(res / fcsv)):
            if dfilt is None or r.get("delta_ms") == dfilt:
                ref[(int(r["seed"]), r["scheme"])] = float(r["mean_phase_s"])
        out = {}
        for s in a.seeds:
            ph = load_phases(fpath(s))
            same = all(k1 == k2 and np.array_equal(x, y) for (k1, x), (k2, y) in zip(ph, gen(s)))
            r = run_all(p, ph)
            row = {"input_equals_generator": same}
            for sch in SCHEMES:
                got, exp = r[sch]["mean_phase_s"], ref[(s, sch)]
                ok = abs(got - exp) <= 1e-6 * max(1.0, abs(exp))
                bad += (not ok) + (not same)
                row[sch] = {"reproduced_s": got, "stored_s": exp, "match": ok}
            out[s] = row
            print(f"[{name}] seed {s}: " + ", ".join(f"{k.split(' ')[0]}={v['reproduced_s']:.4f}{'' if v['match'] else '(!)'}"
                                                     for k, v in row.items() if isinstance(v, dict)), flush=True)
        means = {sch: float(np.mean([out[s][sch]["reproduced_s"] for s in a.seeds])) for sch in SCHEMES}
        tm = np.array([out[s]["SAD-OTR-TM"]["reproduced_s"] for s in a.seeds])
        ecl = np.array([out[s]["Eclipse-K+LP"]["reproduced_s"] for s in a.seeds])
        report[name] = {"seeds": a.seeds, "per_seed": out, "mean_over_seeds_s": means,
                        "TM_over_EclipseLP_mean_ratio": float(np.mean(tm / ecl)),
                        "TM_faster_seeds": int(np.sum(tm < ecl))}
        print(f"[{name}] mean over seeds: " + ", ".join(f"{k}={v:.4f}" for k, v in means.items()))
        print(f"[{name}] TM/Eclipse-K+LP = {report[name]['TM_over_EclipseLP_mean_ratio']:.4f}, TM faster on {report[name]['TM_faster_seeds']}/{len(a.seeds)} seeds")
    od = ROOT / "results_reproduced"; od.mkdir(exist_ok=True)
    (od / "headline_check.json").write_text(json.dumps(report, indent=1, default=str))
    print("ALL MATCH" if bad == 0 else f"{bad} MISMATCHES")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
