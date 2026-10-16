#!/usr/bin/env python3
"""Generate LaTeX rows for the tables added in the final-check revision
(significance, SAD-OTR-TM ablation, model-derived workload) directly from
results/, so that no cell is typed by hand.  Output: results/generated_tables.tex
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
out = []


def pfmt(p):
    if p < 1e-3:
        e = f"{p:.1e}"
        m, x = e.split("e")
        return f"${m}\\times10^{{{int(x)}}}$"
    return f"{p:.3f}"


sig = json.loads((R / "significance.json").read_text())["tests"]
names = {"SAD-OTR-TM": "TM", "Eclipse-K+LP": "Eclipse-K+LP", "Eclipse-K": "Eclipse-K",
         "SAD-OTR (GHEM-AC)": "GHEM-AC", "MWM per phase": "MWM", "RotorNet-style": "RotorNet",
         "Static OCS": "Static OCS", "Solstice-K+LP": "Solstice-K+LP", "Solstice-K": "Solstice-K"}
out.append("% significance")
for t in sig:
    out.append(f"{names[t['a']]} vs {names[t['b']]} & ${t['mean_ratio_a_over_b']:.3f}\\pm{t['ratio_ci95']:.3f}$ & "
               f"{t['wins_a_faster']}/10 & {pfmt(t['p_t_holm'])} & {pfmt(t['p_wilcoxon_holm'])} & "
               f"{t['cohen_dz_log']:.1f} \\\\")

ab = json.loads((R / "tm_ablation.json").read_text())["summary"]
out.append("% tm ablation")
for s in ab:
    out.append(f"{s['variant']} & ${s['mean_phase_s']:.3f}\\pm{s['ci95']:.3f}$ & "
               f"${s['ratio_to_default']:.3f}\\pm{s['ratio_ci95']:.3f}$ & {s['n_reconfig']:.1f} & "
               f"{s['mean_latency_ms']:.1f} ({s['max_latency_ms']:.0f}) \\\\")

llm = json.loads((R / "llm_summary.json").read_text())
out.append("% llm delta=15")
for name, d in llm["summary"]["15.0"].items():
    opt = "--" if d["optical_pct"]["mean"] != d["optical_pct"]["mean"] else f"{d['optical_pct']['mean']:.1f}"
    out.append(f"{name} & ${1000*d['mean_phase_s']['mean']:.1f}\\pm{1000*d['mean_phase_s']['ci95_half']:.1f}$ & "
               f"{d['speedup_vs_eps']['mean']:.2f} & {d['ratio_to_tm']['mean']:.3f} & {opt} & "
               f"{d['n_reconfig']['mean']:.1f} & {d['mean_latency_ms']['mean']:.1f} \\\\")
for dms in ("1.0", "100.0"):
    out.append(f"% llm delta={dms}")
    for name, d in llm["summary"][dms].items():
        out.append(f"%  {name}: {1000*d['mean_phase_s']['mean']:.1f} ms x{d['speedup_vs_eps']['mean']:.2f} rec {d['n_reconfig']['mean']:.1f} /TM {d['ratio_to_tm']['mean']:.3f}")
(R / "generated_tables.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
