#!/usr/bin/env python3
"""Provenance check for the manuscript (final-check revision).

Every decimal number in the manuscript body (abstract to bibliography) is
looked up in the summary-level result files, allowing only identity, x100
(fraction -> percent), 100(v-1) / 100(1-v) (ratio -> percent change) and
1/v (inverse ratio).  Each number is reported with the file and key path that
reproduces it after rounding to the printed precision; numbers with no source
are listed as UNMATCHED for manual review.  A false-match estimate is printed
by testing random numbers with the same precision profile.
Usage: python tools/check_numbers.py path/to/bare_jrnl.tex [--verbose]
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SUMMARY = [
    "main_tables.json", "table_agg.csv", "table_beps.csv", "table_k.csv", "table_opt.csv",
    "latency_table_ii.json", "ablation_split.json", "multiseed.json", "volume_sweep.csv",
    "delta_sweep.csv", "baselines_summary.json", "baselines_phasekind.csv", "baselines_delta.csv",
    "scalability.csv", "scalability.json", "robust.json", "significance.json", "tm_ablation.csv",
    "llm_summary.json", "derived_numbers.json",
]
TRANSFORMS = [
    ("id", lambda v: v), ("x100", lambda v: 100 * v),
    ("pct+", lambda v: 100 * (v - 1)), ("pct-", lambda v: 100 * (1 - v)),
    ("inv", lambda v: 1 / v if v else float("nan")),
]


def harvest():
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        else:
            try:
                f = float(o)
            except (TypeError, ValueError):
                return
            if math.isfinite(f):
                out.append((f, path))

    for name in SUMMARY:
        p = RES / name
        if not p.exists():
            continue
        if name.endswith(".json"):
            walk(json.loads(p.read_text()), name)
        else:
            rows = list(csv.reader(p.open()))
            hdr = rows[0]
            for r_i, row in enumerate(rows[1:]):
                for h, c in zip(hdr, row):
                    try:
                        f = float(c)
                    except ValueError:
                        continue
                    out.append((f, f"{name}[{r_i}].{h}({row[0]}|{row[1] if len(row)>1 else ''})"))
    pool = []
    for v, path in out:
        for tn, fn in TRANSFORMS:
            try:
                w = fn(v)
            except ZeroDivisionError:
                continue
            if math.isfinite(w):
                pool.append((w, f"{tn}:{path}"))
    pool.sort()
    return pool


def lookup(pool, keys, x, dec):
    tol = 0.5 * 10 ** (-dec) + 1e-9
    i = bisect.bisect_left(keys, x - tol)
    hits = []
    while i < len(keys) and keys[i] <= x + tol:
        hits.append(pool[i][1]); i += 1
    return hits


def main():
    tex = Path(sys.argv[1]).read_text()
    verbose = "--verbose" in sys.argv
    body = tex.split("\\begin{abstract}")[1].split("\\begin{thebibliography}")[0]
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\(label|ref|eqref|cite|includegraphics|setlength|begin|end)\{[^}]*\}", " ", body)
    pool = harvest(); keys = [p[0] for p in pool]
    nums = list(re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", body))
    unmatched = []
    for m in nums:
        s = m.group(1); dec = len(s.split(".")[1])
        hits = lookup(pool, keys, float(s), dec)
        ctx = body[max(0, m.start() - 50): m.end() + 25].replace("\n", " ")
        if hits:
            if verbose:
                print(f"  OK {s:>9s} <- {hits[0]}  | {ctx}")
        else:
            unmatched.append((s, ctx))
    rnd = random.Random(0); fm = 0; trials = 2000
    for _ in range(trials):
        m = rnd.choice(nums).group(1); dec = len(m.split(".")[1])
        x = round(rnd.uniform(0, 2 * float(m) + 1), dec)
        fm += bool(lookup(pool, keys, x, dec))
    print(f"decimal numbers checked: {len(nums)}; source pool: {len(pool)} values")
    print(f"estimated false-match rate for random numbers of the same precision: {100*fm/trials:.1f}%")
    print(f"UNMATCHED: {len(unmatched)}")
    for s, c in unmatched:
        print(f"  {s:>9s} | ...{c}...")


if __name__ == "__main__":
    main()
