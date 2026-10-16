#!/usr/bin/env python3
"""Figures added/regenerated in the Q1 revision (reads results/*.csv only,
except fig_workload which draws the demand matrices from the generator).

  fig_workload.pdf    demand matrices of the three phase types (+ skewed EP)
  fig_baselines.pdf   literature baselines with 95% CIs + delta sweep
  fig_scalability.pdf decision latency and quality vs fabric size
  fig_robust.pdf      demand-uncertainty study (seeds 0-9, 95% CIs)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_figures import C, _setup_mpl  # noqa: E402
from sad_otr.demands import make_phase_sequence, make_phase_sequence_skewed  # noqa: E402

RES = ROOT / "results"
FIG = ROOT.parent / "figures"
FIG.mkdir(exist_ok=True)

SHORT = {
    "EPS Clos": "EPS", "Static OCS": "Static\nOCS", "RotorNet-style": "Rotor",
    "MWM per phase": "MWM", "Solstice-K": "Sol.", "Solstice-K+LP": "Sol.\n+LP",
    "Eclipse-K": "Ecl.", "Eclipse-K+LP": "Ecl.\n+LP", "SAD-OTR (GHEM-AC)": "GHEM\n-AC",
    "SAD-OTR-TM": "SAD-\nOTR-TM", "Ideal bound": "Ideal\nbound",
}
COL = {
    "EPS Clos": C["eps"], "Static OCS": C["static"], "RotorNet-style": "#999999",
    "MWM per phase": "#CC79A7", "Solstice-K": "#F0E442", "Solstice-K+LP": "#C9BA2B",
    "Eclipse-K": "#56B4E9", "Eclipse-K+LP": "#2B7FB0", "SAD-OTR (GHEM-AC)": "#7FCDB1",
    "SAD-OTR-TM": C["sad"], "Ideal bound": "#000000",
}


def fig_workload():
    plt = _setup_mpl()
    ph = make_phase_sequence(32, 12, 0)
    sk = make_phase_sequence_skewed(32, 12, 0)
    mats = [("(a) TP ring", ph[0][1]), ("(b) EP, fan-out 4", ph[1][1]),
            ("(c) PP chain", ph[2][1]), ("(d) skewed EP (stress)", sk[1][1])]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))
    vmax = max(float(M.max()) for _, M in mats)
    for ax, (t, M) in zip(axes, mats):
        im = ax.imshow(np.where(M > 0, M, np.nan), cmap="viridis", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(t, fontsize=8)
        ax.set_xticks([0, 15, 31]); ax.set_yticks([0, 15, 31])
        ax.set_xlabel("Destination rack", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_facecolor("#f4f4f4")
    axes[0].set_ylabel("Source rack", fontsize=7)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Gb", fontsize=7); cb.ax.tick_params(labelsize=6)
    out = FIG / "fig_workload.pdf"
    fig.savefig(out); plt.close(fig); print("Wrote", out)


def fig_baselines():
    plt = _setup_mpl()
    summ = json.loads((RES / "baselines_summary.json").read_text())["summary"]
    order = ["EPS Clos", "Static OCS", "RotorNet-style", "MWM per phase", "Solstice-K",
             "Solstice-K+LP", "Eclipse-K", "Eclipse-K+LP", "SAD-OTR (GHEM-AC)", "SAD-OTR-TM"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(order))
    m = [summ[n]["mean_phase_s"]["mean"] for n in order]
    h = [summ[n]["mean_phase_s"]["ci95_half"] for n in order]
    a0.bar(x, m, 0.7, yerr=h, capsize=2, color=[COL[n] for n in order], edgecolor="none",
           error_kw={"elinewidth": 0.8})
    lb = summ["Ideal bound"]["mean_phase_s"]["mean"]
    a0.axhline(lb, color="k", linestyle="--", linewidth=0.9)
    a0.text(-0.4, lb + 0.04, "ideal bound", ha="left", va="bottom", fontsize=6)
    for xi, mi in zip(x, m):
        a0.text(xi, mi + 0.08, f"{mi:.2f}", ha="center", va="bottom", fontsize=6)
    a0.set_xticks(x); a0.set_xticklabels([SHORT[n] for n in order], fontsize=6)
    a0.set_ylabel("Mean phase time (s)")
    a0.set_title("(a) $N{=}32$, $\\delta{=}15$ ms, seeds 0-9 (95% CI)")
    a0.spines["top"].set_visible(False); a0.spines["right"].set_visible(False)
    a0.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7); a0.set_axisbelow(True)

    rows = list(csv.DictReader((RES / "baselines_delta.csv").open()))
    ds = ["EPS Clos", "Static OCS", "MWM per phase", "Solstice-K+LP", "Eclipse-K", "Eclipse-K+LP",
          "SAD-OTR (GHEM-AC)", "SAD-OTR-TM", "Ideal bound"]
    mk = {"EPS Clos": "o", "Static OCS": "s", "MWM per phase": "v", "Solstice-K+LP": "P",
          "Eclipse-K": "x", "Eclipse-K+LP": "D", "SAD-OTR (GHEM-AC)": "^", "SAD-OTR-TM": "*",
          "Ideal bound": ""}
    for n in ds:
        rr = [r for r in rows if r["scheme"] == n]
        dx = [float(r["delta_ms"]) for r in rr]
        dy = [float(r["mean_phase_s"]) for r in rr]
        de = [float(r["ci95_half"]) for r in rr]
        a1.errorbar(dx, dy, yerr=de, marker=mk[n], markersize=3.5, color=COL[n],
                    linestyle="--" if n == "Ideal bound" else "-", linewidth=1.1, capsize=1.5,
                    label=SHORT[n].replace("\n", " ").replace("- ", "-"))
    a1.set_xscale("log")
    a1.axvspan(10, 20, color="0.88", alpha=0.7)
    a1.set_xlabel(r"Reconfiguration delay $\delta$ (ms)")
    a1.set_ylabel("Mean phase time (s)")
    a1.set_title("(b) $\\delta$ sweep, seeds 0-4")
    a1.set_ylim(0.3, 4.3)
    a1.legend(frameon=False, fontsize=5.3, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.6)
    a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)
    a1.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout(w_pad=1.2)
    out = FIG / "fig_baselines.pdf"
    fig.savefig(out); plt.close(fig); print("Wrote", out)


def fig_scalability():
    plt = _setup_mpl()
    rows = list(csv.DictReader((RES / "scalability.csv").open()))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for fam, ls in (("uniform", "-"), ("skewed", "--")):
        rr = [r for r in rows if r["family"] == fam]
        Ns = [int(r["N"]) for r in rr]
        a0.loglog(Ns, [float(r["ghem_mean_ms"]) for r in rr], "o" + ls, color=C["ghem"], markersize=4,
                  label=f"GHEM-AC ({fam})")
        a0.loglog(Ns, [float(r["milp_median_ms"]) for r in rr], "s" + ls, color=C["milp"], markersize=4,
                  label=f"MILP median ({fam})")
        a0.loglog(Ns, [float(r["tm_mean_ms"]) for r in rr], "*" + ls, color=C["sad"], markersize=5,
                  label=f"SAD-OTR-TM ({fam})")
    a0.axhline(15, color="0.5", linestyle=":", linewidth=0.9)
    a0.text(16, 17, r"$\delta=15$ ms", fontsize=7, color="0.4")
    from matplotlib.ticker import NullFormatter
    a0.set_xticks([16, 32, 64, 128, 256]); a0.set_xticklabels(["16", "32", "64", "128", "256"])
    a0.xaxis.set_minor_formatter(NullFormatter())
    a0.set_xlabel("Number of racks $N$"); a0.set_ylabel("Decision latency (ms)")
    a0.set_title("(a) Decision latency (Python, 1 thread)")
    a0.legend(frameon=False, fontsize=5.5, ncol=1, loc="upper left")
    a0.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    a0.spines["top"].set_visible(False); a0.spines["right"].set_visible(False)
    for fam, ls in (("uniform", "-"), ("skewed", "--")):
        rr = [r for r in rows if r["family"] == fam]
        Ns = [int(r["N"]) for r in rr]
        g = [float(r["mean_gap_pct"]) if r["mean_gap_pct"] not in ("", "None") else np.nan for r in rr]
        w = [float(r["worst_gap_pct"]) if r["worst_gap_pct"] not in ("", "None") else np.nan for r in rr]
        tl = [100 * (float(r["tm_over_lb"]) - 1) for r in rr]
        a1.plot(Ns, g, "o" + ls, color=C["mean"], markersize=4, label=f"GHEM mean gap ({fam})")
        a1.plot(Ns, w, "s" + ls, color=C["worst"], markersize=4, label=f"GHEM worst gap ({fam})")
        a1.plot(Ns, tl, "*" + ls, color=C["sad"], markersize=5, label=f"TM above ideal ({fam})")
    a1.set_xscale("log", base=2)
    a1.set_xticks([16, 32, 64, 128, 256]); a1.set_xticklabels(["16", "32", "64", "128", "256"])
    a1.set_xlabel("Number of racks $N$"); a1.set_ylabel("Percent")
    a1.set_title("(b) Quality: gap to MILP / to ideal bound")
    a1.set_ylim(-0.5, 19)
    a1.legend(frameon=False, fontsize=5.3, loc="upper center", ncol=2, columnspacing=0.8)
    a1.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    out = FIG / "fig_scalability.pdf"
    fig.savefig(out); plt.close(fig); print("Wrote", out)


def fig_robust():
    plt = _setup_mpl()
    rows = list(csv.DictReader((RES / "robust.csv").open()))
    base = [r for r in rows if r["mode"] == "volume" and float(r["level"]) == 0.0][0]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, mode, xl in ((a0, "volume", "Volume error $f$ (entries scaled by U[1-f,1+f])"),
                         (a1, "fanout", "EP destination mis-prediction probability $p$")):
        rr = [base] + [r for r in rows if r["mode"] == mode and float(r["level"]) > 0] if mode == "fanout" \
            else [r for r in rows if r["mode"] == mode]
        xs = [float(r["level"]) for r in rr]
        ax.errorbar(xs, [float(r["ghem_abs_s"]) for r in rr], yerr=[float(r["ghem_abs_ci95"]) for r in rr],
                    marker="^", color=COL["SAD-OTR (GHEM-AC)"], capsize=2, label="SAD-OTR (GHEM-AC)")
        ax.errorbar(xs, [float(r["tm_abs_s"]) for r in rr], yerr=[float(r["tm_abs_ci95"]) for r in rr],
                    marker="*", color=COL["SAD-OTR-TM"], capsize=2, label="SAD-OTR-TM")
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("Mean phase time on true $D$ (s)")
        ax.set_ylim(0, None)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=7)
    a0.set_title("(a) Volume error"); a1.set_title("(b) Expert fan-out error")
    fig.tight_layout(w_pad=1.5)
    out = FIG / "fig_robust.pdf"
    fig.savefig(out); plt.close(fig); print("Wrote", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["workload", "baselines", "scalability", "robust"]
    for w in which:
        globals()[f"fig_{w}"]()
