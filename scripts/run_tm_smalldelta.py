#!/usr/bin/env python3
"""Why is Eclipse-K+LP faster than SAD-OTR-TM at delta=0.1 ms?  Re-run TM at
delta=0.1 ms (seeds 0-4, as in the delta sweep) with the default slice cap
(32), a 4x larger cap (128), and without the GHEM-AC first slice.
Output: results/tm_smalldelta.json
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sad_otr.demands import make_phase_sequence
from sad_otr.multislot import sad_otr_tm_variant
N, K, BO, BE, D0 = 32, 4, 800.0, 400.0, 0.0001
out = {}
for name, kw in (("default L_max=32", {}), ("L_max=128", {"n_slices": 128}),
                 ("no GHEM-AC first slice, L_max=32", {"first_slice": "greedy"}),
                 ("no GHEM-AC first slice, L_max=128", {"first_slice": "greedy", "n_slices": 128})):
    per_seed = []
    for seed in range(5):
        E = np.zeros((N, N), dtype=np.int8); Ts = []; rec = 0; capped = 0
        for _, D in make_phase_sequence(N, 12, seed):
            o = sad_otr_tm_variant(D, E, K, BO, BE, D0, **kw)
            Ts.append(o["T"]); rec += o["n_charged"]; E = o["E_last"]
            capped += int(o["n_generated"] >= kw.get("n_slices", 32))
        per_seed.append({"seed": seed, "mean_phase_s": float(np.mean(Ts)), "n_reconfig": rec, "phases_at_cap": capped})
    out[name] = {"mean_phase_s": float(np.mean([r["mean_phase_s"] for r in per_seed])),
                 "n_reconfig": float(np.mean([r["n_reconfig"] for r in per_seed])),
                 "phases_at_cap": float(np.mean([r["phases_at_cap"] for r in per_seed])), "per_seed": per_seed}
    print(name, {k: v for k, v in out[name].items() if k != "per_seed"}, flush=True)
(ROOT / "results" / "tm_smalldelta.json").write_text(json.dumps(out, indent=2))
