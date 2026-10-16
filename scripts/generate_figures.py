#!/usr/bin/env python3
"""Generate IEEE-style PDF figures for the SAD-OTR manuscript.

Reads existing results under supplement/results/ and runs additional
evaluations for phases.csv, volume_sweep.csv, and delta_sweep.csv.
Does not invent numbers — all plotted values come from the simulator
or from previously written results JSON/CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sad_otr.demands import make_phase_sequence  # noqa: E402
from sad_otr.evaluate import FabricParams, run_scheme  # noqa: E402

RESULTS = ROOT / "results"
FIGDIR = ROOT.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# Nominal TP ring volume in make_phase_sequence / tensor_ring default.
REF_TP_GB = 200.0


def scale_default_phases(
    vol_gb: float, N: int = 32, n_phases: int = 12, seed: int = 0
) -> list[tuple[str, np.ndarray]]:
    """Scale the default 12-phase suite so the TP ring volume equals vol_gb.

    Preserves EP/PP relative volumes from Section V. At vol_gb=200 the suite
    matches the Table I operating point (speedup ≈ 2.49).
    """
    scale = float(vol_gb) / REF_TP_GB
    base = make_phase_sequence(N, n_phases, seed)
    return [(k, D * scale) for k, D in base]


def export_phases_csv() -> Path:
    """Per-phase completion times for EPS / Static OCS / SAD-OTR (N=32, seed=0)."""
    params = FabricParams(N=32, K=4, seed=0, n_phases=12)
    phases = make_phase_sequence(params.N, params.n_phases, params.seed)
    schemes = {
        "eps": "EPS Clos",
        "static_ocs": "Static OCS",
        "ghem": "SAD-OTR (GHEM-AC)",
    }
    rows = []
    for key, label in schemes.items():
        r = run_scheme(params, scheme=key, phases=phases, use_repair=False)
        for rec in r["phases"]:
            rows.append({
                "phase": rec["phase"],
                "kind": rec["kind"],
                "scheme": label,
                "T_s": round(float(rec["T"]), 6),
                "reconfigured": int(bool(rec["reconfigured"])),
            })
        print(
            f"phases {label}: mean={r['mean_phase_s']:.4f}s "
            f"n_reconfig={r['n_reconfig']}"
        )

    out = RESULTS / "phases.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["phase", "kind", "scheme", "T_s", "reconfigured"]
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out)
    return out


def run_volume_sweep() -> Path:
    """Sweep transfer size 0.1–200 Gb/pair; delta=15 ms, N=32, K=4."""
    vols = np.unique(
        np.round(
            np.concatenate([
                np.geomspace(0.1, 200.0, 18),
                np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]),
            ]),
            decimals=6,
        )
    )
    vols = np.sort(vols)
    params = FabricParams(N=32, K=4, delta=0.015, seed=0, n_phases=12)
    rows = []
    for v in vols:
        ph = scale_default_phases(float(v), params.N, params.n_phases, params.seed)
        e = run_scheme(params, scheme="eps", phases=ph)
        g = run_scheme(params, scheme="ghem", phases=ph, use_repair=False)
        speedup = e["mean_phase_s"] / g["mean_phase_s"] if g["mean_phase_s"] > 0 else float("inf")
        row = {
            "vol_gb": float(v),
            "eps_mean_s": round(e["mean_phase_s"], 6),
            "sad_mean_s": round(g["mean_phase_s"], 6),
            "speedup_vs_eps": round(speedup, 4),
            "n_reconfig": int(g["n_reconfig"]),
            "scale_vs_default": round(float(v) / REF_TP_GB, 8),
        }
        rows.append(row)
        print(
            f"vol={v:8.3f} Gb: speedup={row['speedup_vs_eps']:.3f} "
            f"recon={row['n_reconfig']}"
        )

    out = RESULTS / "volume_sweep.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out)
    return out


def run_delta_sweep() -> Path:
    """Sweep delta from 1 ms to 1 s at full (default) transfer sizes."""
    deltas_s = np.unique(
        np.round(
            np.concatenate([
                np.geomspace(0.001, 1.0, 16),
                np.array([0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]),
            ]),
            decimals=6,
        )
    )
    deltas_s = np.sort(deltas_s)
    phases = make_phase_sequence(32, 12, 0)
    rows = []
    for d in deltas_s:
        p = FabricParams(N=32, K=4, delta=float(d), seed=0, n_phases=12)
        g = run_scheme(p, scheme="ghem", phases=phases, use_repair=False)
        row = {
            "delta_s": float(d),
            "delta_ms": round(float(d) * 1000.0, 6),
            "mean_phase_s": round(g["mean_phase_s"], 6),
            "n_reconfig": int(g["n_reconfig"]),
        }
        rows.append(row)
        print(
            f"delta={d*1000:8.3f} ms: mean={row['mean_phase_s']:.4f}s "
            f"recon={row['n_reconfig']}"
        )

    out = RESULTS / "delta_sweep.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_mpl():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


# Colorblind-friendly Okabe–Ito palette
C = {
    "eps": "#0072B2",       # blue
    "static": "#E69F00",    # orange
    "sad": "#009E73",       # bluish green
    "ghem": "#56B4E9",      # sky blue
    "milp": "#D55E00",      # vermillion
    "mean": "#0072B2",
    "worst": "#D55E00",
    "reconfig": "#CC79A7",  # reddish purple
    "speedup": "#009E73",
}


def plot_phases(csv_path: Path) -> Path:
    plt = _setup_mpl()
    rows = list(csv.DictReader(csv_path.open()))
    by_scheme: dict[str, list] = {}
    kinds = {}
    for r in rows:
        by_scheme.setdefault(r["scheme"], []).append(float(r["T_s"]))
        kinds[int(r["phase"])] = r["kind"]
    n = len(kinds)
    x = np.arange(n)
    width = 0.26
    order = ["EPS Clos", "Static OCS", "SAD-OTR (GHEM-AC)"]
    colors = [C["eps"], C["static"], C["sad"]]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i, (sch, col) in enumerate(zip(order, colors)):
        ys = by_scheme[sch]
        ax.bar(x + (i - 1) * width, ys, width, label=sch, color=col, edgecolor="none")

    ax.set_xticks(x)
    labels = [f"{i}\n{kinds[i]}" for i in range(n)]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Phase index (TP / EP / PP)")
    ax.set_ylabel("Completion time (s)")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.set_xlim(-0.55, n - 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    out = FIGDIR / "fig_phases.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)
    return out


def plot_optimality() -> Path:
    plt = _setup_mpl()
    # Prefer results JSON; Table II latency from latency_table_ii.json (medians match ms)
    main = json.loads((RESULTS / "main_tables.json").read_text())
    opt = main["table_opt"]
    lat = json.loads((RESULTS / "latency_table_ii.json").read_text())["table_opt_latency"]

    Ns = [r["N"] for r in opt]
    mean_gap = [r["mean_gap_pct"] for r in opt]
    worst_gap = [r["worst_gap_pct"] for r in opt]

    # Prefer latency_table_ii medians for MILP / means for GHEM (match Table II caption)
    lat_by_n = {r["N"]: r for r in lat}
    ghem_ms = [lat_by_n[n]["ghem_mean_ms"] for n in Ns]
    milp_ms = [lat_by_n[n]["milp_median_ms"] for n in Ns]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax0.plot(Ns, mean_gap, "o-", color=C["mean"], label="Mean gap", markersize=5)
    ax0.plot(Ns, worst_gap, "s--", color=C["worst"], label="Worst gap", markersize=5)
    ax0.set_xlabel("Number of racks $N$")
    ax0.set_ylabel("Optimality gap (\\%)")
    ax0.set_ylim(-0.05, 1.0)
    ax0.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax0.legend(frameon=False, loc="upper right")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax0.set_title("(a) Gap vs.\\ MILP")
    # Fix ylabel - use plain % without latex backslash issues in matplotlib
    ax0.set_ylabel("Optimality gap (%)")
    ax0.set_title("(a) Gap vs. MILP")

    ax1.plot(Ns, ghem_ms, "o-", color=C["ghem"], label="GHEM-AC (mean)", markersize=5)
    ax1.plot(Ns, milp_ms, "s--", color=C["milp"], label="MILP (median)", markersize=5)
    ax1.set_xlabel("Number of racks $N$")
    ax1.set_ylabel("Decision latency (ms)")
    ax1.legend(frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.set_title("(b) Decision latency")

    fig.tight_layout(w_pad=2.0)
    out = FIGDIR / "fig_optimality.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)
    return out


def plot_volume(csv_path: Path) -> Path:
    plt = _setup_mpl()
    rows = list(csv.DictReader(csv_path.open()))
    vols = np.array([float(r["vol_gb"]) for r in rows])
    speedup = np.array([float(r["speedup_vs_eps"]) for r in rows])
    n_rec = np.array([int(r["n_reconfig"]) for r in rows])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax0.semilogx(vols, speedup, "o-", color=C["speedup"], markersize=4)
    ax0.axvline(1.0, color="0.5", linestyle=":", linewidth=0.9)
    ax0.set_xlabel("Transfer size per active pair (Gb)")
    ax0.set_ylabel("Speedup vs. EPS Clos")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax0.set_title("(a) Speedup")
    ax0.set_xlim(vols.min() * 0.9, vols.max() * 1.1)

    ax1.semilogx(vols, n_rec, "s-", color=C["reconfig"], markersize=4)
    ax1.axvline(1.0, color="0.5", linestyle=":", linewidth=0.9)
    ax1.set_xlabel("Transfer size per active pair (Gb)")
    ax1.set_ylabel("Reconfigurations adopted")
    ax1.set_ylim(-0.5, max(12, n_rec.max()) + 0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.set_title("(b) Reconfig. count")
    ax1.set_xlim(vols.min() * 0.9, vols.max() * 1.1)

    fig.tight_layout(w_pad=2.0)
    out = FIGDIR / "fig_volume.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)
    return out


def plot_delta(csv_path: Path) -> Path:
    plt = _setup_mpl()
    rows = list(csv.DictReader(csv_path.open()))
    d_ms = np.array([float(r["delta_ms"]) for r in rows])
    mean_t = np.array([float(r["mean_phase_s"]) for r in rows])
    n_rec = np.array([int(r["n_reconfig"]) for r in rows])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax0.semilogx(d_ms, mean_t, "o-", color=C["sad"], markersize=4)
    ax0.axvspan(10, 20, color="0.85", alpha=0.6, label="Deployed mirror range")
    ax0.set_xlabel(r"Reconfiguration delay $\delta$ (ms)")
    ax0.set_ylabel("Mean phase time (s)")
    ax0.legend(frameon=False, loc="upper left")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax0.set_title("(a) Mean phase time")

    ax1.semilogx(d_ms, n_rec, "s-", color=C["reconfig"], markersize=4)
    ax1.axvspan(10, 20, color="0.85", alpha=0.6)
    ax1.set_xlabel(r"Reconfiguration delay $\delta$ (ms)")
    ax1.set_ylabel("Reconfigurations adopted")
    ax1.set_ylim(-0.5, max(12, n_rec.max()) + 0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.set_title("(b) Reconfig. count")

    fig.tight_layout(w_pad=2.0)
    out = FIGDIR / "fig_delta.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)
    return out


def plot_sensitivity() -> Path:
    plt = _setup_mpl()
    k_rows = list(csv.DictReader((RESULTS / "table_k.csv").open()))
    b_rows = list(csv.DictReader((RESULTS / "table_beps.csv").open()))

    Ks = [int(r["K"]) for r in k_rows]
    speedup = [float(r["speedup"]) for r in k_rows]
    Beps = [float(r["B_eps"]) for r in b_rows]
    eps_s = [float(r["eps_s"]) for r in b_rows]
    static_s = [float(r["static_s"]) for r in b_rows]
    sad_s = [float(r["sad_s"]) for r in b_rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax0.plot(Ks, speedup, "o-", color=C["speedup"], markersize=5)
    ax0.set_xlabel(r"Optical ports per rack $K$")
    ax0.set_ylabel("Speedup vs. EPS Clos")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax0.set_title("(a) Port count")
    ax0.set_xticks(Ks)

    ax1.plot(Beps, eps_s, "o-", color=C["eps"], label="EPS Clos", markersize=5)
    ax1.plot(Beps, static_s, "s--", color=C["static"], label="Static OCS", markersize=5)
    ax1.plot(Beps, sad_s, "^-", color=C["sad"], label="SAD-OTR", markersize=5)
    ax1.set_xlabel(r"Packet capacity $B_{\mathrm{eps}}$ (Gbps)")
    ax1.set_ylabel("Mean phase time (s)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(Beps)
    ax1.set_xticklabels([str(int(b)) for b in Beps])
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.set_title("(b) Fallback capacity")

    fig.tight_layout(w_pad=2.0)
    out = FIGDIR / "fig_sensitivity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)
    return out


def write_readme(paths: dict[str, Path]) -> Path:
    text = """# SAD-OTR manuscript figures

IEEE-style vector PDFs generated from the supplementary simulator under
`../supplement/`. Do **not** invent numbers; all values come from
`../supplement/results/*.json|*.csv` or from fresh `run_scheme` evaluations.

## How each figure was produced

| File | Source | Method |
|------|--------|--------|
| `fig_phases.pdf` | `../supplement/results/phases.csv` | Run EPS Clos, Static OCS, SAD-OTR (GHEM-AC) on N=32, seed=0, 12-phase suite via `sad_otr.evaluate.run_scheme`; grouped bars by phase with TP/EP/PP labels. |
| `fig_optimality.pdf` | `main_tables.json` (`table_opt`) + `latency_table_ii.json` | Left: mean/worst optimality gap (%) vs N. Right: GHEM-AC mean latency and MILP median wall time (ms) vs N (Table II numbers from latency JSON). |
| `fig_volume.pdf` | `../supplement/results/volume_sweep.csv` | Sweep transfer size 0.1–200 Gb (scale default suite so TP ring = vol_gb; EP/PP keep Section V ratios), δ=15 ms, N=32, K=4; speedup vs EPS and reconfig count. At 200 Gb recovers Table I (~2.49×). |
| `fig_delta.pdf` | `../supplement/results/delta_sweep.csv` | Sweep δ from 1 ms to 1 s at full default transfer sizes; mean phase time and reconfig count. |
| `fig_sensitivity.pdf` | `table_k.csv`, `table_beps.csv` | Left: speedup vs K. Right: mean phase time vs B_eps for EPS / Static OCS / SAD-OTR. |

## Not generated here

- **`fig_arch.pdf`** — system architecture diagram. Keep the existing Overleaf
  asset; this script does not invent a fake architecture figure. Re-download
  from Overleaf if missing locally.

## Regenerate

```bash
cd ../supplement
source .venv/bin/activate
python scripts/generate_figures.py
```

Outputs land in this directory (`figures/`) and CSVs under `../supplement/results/`.
"""
    out = FIGDIR / "README.md"
    out.write_text(text)
    print("Wrote", out)
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    phases_csv = export_phases_csv()
    vol_csv = run_volume_sweep()
    delta_csv = run_delta_sweep()

    paths = {
        "phases": plot_phases(phases_csv),
        "optimality": plot_optimality(),
        "volume": plot_volume(vol_csv),
        "delta": plot_delta(delta_csv),
        "sensitivity": plot_sensitivity(),
    }
    write_readme(paths)
    for name, p in paths.items():
        print(f"OK {name}: {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
