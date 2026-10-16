"""End-to-end evaluation helpers for SAD-OTR schemes and baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .closed_form import optical_byte_share, phase_time, phase_time_unsplittable
from .demands import make_phase_sequence
from .ghem_ac import ghem_ac
from .matching import weight_greedy
from .milp import solve_exact


@dataclass
class FabricParams:
    N: int = 32
    K: int = 4
    B_opt: float = 800.0
    B_eps: float = 400.0
    delta: float = 0.015  # 15 ms
    n_phases: int = 12
    seed: int = 0


def eps_clos_time(D: np.ndarray, B_eps: float) -> float:
    """Static packet Clos: empty optical matching."""
    N = D.shape[0]
    E = np.zeros((N, N), dtype=np.int8)
    return phase_time(D, E, B_opt=0.0, B_eps=B_eps)


def static_ocs_topology(phases: list[tuple[str, np.ndarray]], K: int) -> np.ndarray:
    """Static OCS baseline: WeightGreedy on phase-1 demand, held fixed."""
    _, D0 = phases[0]
    return weight_greedy(D0, K)


def run_scheme(
    params: FabricParams,
    scheme: str = "ghem",
    unsplittable: bool = False,
    use_repair: bool = False,
    phases: list[tuple[str, np.ndarray]] | None = None,
    milp_time_limit: float | None = 120.0,
) -> dict[str, Any]:
    """Run a named scheme over the phase sequence.

    scheme in {'eps', 'static_ocs', 'ghem', 'ghem_delta0', 'milp'}.
    """
    if phases is None:
        phases = make_phase_sequence(params.N, params.n_phases, params.seed)

    N = params.N
    E_prev = np.zeros((N, N), dtype=np.int8)
    static_E = None
    if scheme == "static_ocs":
        static_E = static_ocs_topology(phases, params.K)

    records = []
    total_opt = 0.0
    total_bytes = 0.0
    n_reconfig = 0
    times = []
    latencies = []

    for m, (kind, D) in enumerate(phases):
        if scheme == "eps":
            T = eps_clos_time(D, params.B_eps)
            E = np.zeros((N, N), dtype=np.int8)
            reconfig = False
            opt_b, tot_b = 0.0, float(D.sum())
            lat = 0.0
            extra = {"winner": "EPS"}
        elif scheme == "static_ocs":
            assert static_E is not None
            E = static_E
            if unsplittable:
                T = phase_time_unsplittable(D, E, params.B_opt, params.B_eps)
                opt_b, tot_b = float(D[E > 0].sum()), float(D.sum())
            else:
                T = phase_time(D, E, params.B_opt, params.B_eps)
                opt_b, tot_b = optical_byte_share(D, E, params.B_opt, T)
            reconfig = False
            lat = 0.0
            extra = {"winner": "StaticOCS"}
        elif scheme in ("ghem", "ghem_delta0"):
            delta = 0.0 if scheme == "ghem_delta0" else params.delta
            out = ghem_ac(
                D, E_prev, params.K, params.B_opt, params.B_eps, delta,
                unsplittable=unsplittable, use_repair=use_repair,
            )
            E = out["E"]
            T = out["T_penalized"]
            reconfig = out["reconfigured"]
            opt_b, tot_b = out["opt_bytes"], out["total_bytes"]
            lat = out["latency_s"]
            extra = {
                "winner": out["winner"],
                "repaired": out["repaired"],
                "candidate_scores": out["candidate_scores"],
                "T_comm": out["T_comm"],
            }
            E_prev = E.copy()
        elif scheme == "milp":
            sol = solve_exact(
                D, params.K, params.B_opt, params.B_eps, params.delta,
                E_prev=E_prev, unsplittable=unsplittable,
                time_limit=milp_time_limit,
            )
            E = sol["E"]
            T = sol["objective"]
            reconfig = bool(sol["Delta"])
            opt_b, tot_b = sol["opt_bytes"], sol["total_bytes"]
            lat = sol["wall_s"]
            extra = {
                "status": sol["status"],
                "T_comm": sol["T"],
                "build_s": sol["build_s"],
                "solve_s": sol["solve_s"],
            }
            E_prev = E.copy()
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        if reconfig:
            n_reconfig += 1
        total_opt += opt_b
        total_bytes += tot_b
        times.append(T)
        latencies.append(lat)
        records.append({
            "phase": m,
            "kind": kind,
            "T": T,
            "reconfigured": reconfig,
            "opt_bytes": opt_b,
            "total_bytes": tot_b,
            "latency_s": lat,
            **extra,
        })

    mean_T = float(np.mean(times)) if times else 0.0
    max_T = float(np.max(times)) if times else 0.0
    optical_pct = 100.0 * total_opt / total_bytes if total_bytes > 0 else 0.0
    return {
        "scheme": scheme,
        "params": asdict(params),
        "unsplittable": unsplittable,
        "use_repair": use_repair,
        "mean_phase_s": mean_T,
        "max_phase_s": max_T,
        "optical_pct": optical_pct,
        "n_reconfig": n_reconfig,
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "phases": records,
    }
