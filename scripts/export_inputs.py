#!/usr/bin/env python3
"""Export every generated input that feeds the paper's experiments.

The simulator generates workloads on the fly from (generator, N, seed); this
script writes the exact matrices to inputs/ so that they can be inspected or
fed back (scripts/reproduce_headline.py --inputs).  It also writes
inputs/config.json (fabric parameters, generator parameters, seeds per
experiment) and inputs/MANIFEST.csv (file, sha256, description), and checks
that re-generation is bit-identical to the exported file.

Usage: python scripts/export_inputs.py [OUTDIR]   (default: ../inputs)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sad_otr.demands import make_phase_sequence, make_phase_sequence_skewed, perturb_demand  # noqa: E402
from sad_otr.evaluate import FabricParams  # noqa: E402
from sad_otr.inputs_io import load_phases, save_phases  # noqa: E402
from sad_otr.llm_workload import LLM_CONFIG, make_llm_phase_sequence  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "inputs"
man = []


def put(rel, phases, desc, **meta):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    save_phases(p, phases, **meta)
    back = load_phases(p)
    assert all(k1 == k2 and np.array_equal(a, b) for (k1, a), (k2, b) in zip(phases, back))
    man.append((rel, hashlib.sha256(p.read_bytes()).hexdigest(), desc))


def main():
    # primary workload, N=32, seeds 0-9 (Tables agg/baselines/sig/tmabl, Figs. 5-6, robustness oracle)
    for s in range(10):
        put(f"primary/primary_N32_seed{s}.npz", make_phase_sequence(32, 12, s),
            f"primary workload N=32 seed {s}", generator="make_phase_sequence", N=32, seed=s)
    # optimality table N=8..28 seed 0 (N=32 seed 0 above)
    for N in (8, 12, 16, 20, 24, 28):
        put(f"primary/primary_N{N}_seed0.npz", make_phase_sequence(N, 12, 0),
            f"primary workload N={N} seed 0 (Table opt)", generator="make_phase_sequence", N=N, seed=0)
    # scalability families
    for fam in ("uniform", "skewed"):
        for N in (16, 32, 64, 128, 256):
            for s in (range(3) if N == 256 else range(5)):
                ph = make_phase_sequence(N, 12, s) if fam == "uniform" else make_phase_sequence_skewed(N, 12, s, fanout=8, zipf_s=1.0)
                put(f"scalability/{fam}_N{N}_seed{s}.npz", ph, f"scalability {fam} N={N} seed {s}",
                    generator=("make_phase_sequence" if fam == "uniform" else "make_phase_sequence_skewed"), N=N, seed=s)
    # model-derived (DeepSeek-V3 dimensions) workload
    for s in range(10):
        put(f"llm/llm_seed{s}.npz", make_llm_phase_sequence(s), f"DeepSeek-V3-derived workload seed {s}",
            generator="make_llm_phase_sequence", seed=s)
    # demand-uncertainty estimates D_hat exactly as in scripts/run_robust.py
    settings = [("volume", f) for f in (0.1, 0.2, 0.3, 0.4, 0.5)] + [("fanout", p) for p in (0.1, 0.2, 0.3, 0.5)]
    for mode, lvl in settings:
        for sd in range(10):
            ph = make_phase_sequence(32, 12, sd)
            rng = np.random.default_rng(1000 * sd + int(lvl * 100) + (0 if mode == "volume" else 500))
            hats = []
            for kind, D in ph:
                if mode == "volume":
                    hats.append((kind, perturb_demand(D, lvl, rng, mode="volume")))
                else:
                    hats.append((kind, perturb_demand(D, lvl, rng, mode="fanout") if kind == "EP" else D))
            put(f"robust/estimate_{mode}_{lvl:.1f}_seed{sd}.npz", hats,
                f"perturbed estimate D_hat ({mode}, level {lvl}) for primary seed {sd}", mode=mode, level=lvl, seed=sd)

    fp = asdict(FabricParams())
    cfg = {
        "fabric_default": fp,
        "units": {"demand": "Gb per (src,dst) rack pair", "B_opt/B_eps": "Gbps", "delta": "s"},
        "primary_generator": {"TP": "ring i->i+1, 200 Gb", "EP": "4 distinct uniform destinations, U[100,400] Gb",
                              "PP": "stage chain i->i+1, 800 Gb", "cycle": "TP,EP,PP x4", "n_phases": 12},
        "skewed_generator": {"EP": "8 destinations, Zipf(1.0) popularity, U[100,400] Gb", "other": "as primary"},
        "llm_config": LLM_CONFIG,
        "seeds": {
            "tab:agg, tab:split, tab:k, tab:beps, volume and delta sweeps (seed 0)": [0],
            "tab:baselines, tab:sig, tab:tmabl, fig_baselines(a), fig_robust": list(range(10)),
            "fig_baselines(b) delta sweep, tm_smalldelta": list(range(5)),
            "tab:opt": {"N": [8, 12, 16, 20, 24, 28, 32], "seed": [0]},
            "tab:scal, fig_scalability": {"N<=128": list(range(5)), "N=256": list(range(3))},
            "tab:llm (delta=15 ms)": list(range(10)), "llm delta 1 and 100 ms": list(range(5)),
        },
        "sweeps": {"delta_ms_baselines": [0.1, 1, 5, 15, 50, 100, 200, 500, 1000],
                   "K": [1, 2, 4, 6, 8, 12, 16], "B_eps_gbps": [100, 200, 400, 800, 1600],
                   "robust_volume_f": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "robust_fanout_p": [0.1, 0.2, 0.3, 0.5],
                   "robust_rng_seed": "1000*seed + int(100*level) + (0 if volume else 500)"},
        "solvers": {"MILP": "CBC via PuLP, gap 0, 1 thread, 120 s (60 s scalability)", "LP": "HiGHS via SciPy"},
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    man.append(("config.json", hashlib.sha256((OUT / "config.json").read_bytes()).hexdigest(), "parameters, generators, seeds"))
    with (OUT / "MANIFEST.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "sha256", "description"]); w.writerows(man)
    print(f"exported {len(man)} files to {OUT}")


if __name__ == "__main__":
    main()
