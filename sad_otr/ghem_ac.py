"""GHEM-AC: three candidates + amortization; optional LocalSwapRepair."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .closed_form import (
    optical_byte_share,
    optical_byte_share_unsplittable,
    phase_time,
    phase_time_unsplittable,
)
from .matching import bisect_target, bottleneck_fill, local_swap_repair, weight_greedy


def ghem_ac(
    D: np.ndarray,
    E_prev: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    unsplittable: bool = False,
    use_repair: bool = False,
) -> dict[str, Any]:
    """One GHEM-AC decision.

    If use_repair=True, LocalSwapRepair is applied to the best of the three
    candidates before the amortization test (documented 4th/post-repair path).
    """
    t0 = time.perf_counter()

    E1 = weight_greedy(D, K)
    E2 = bottleneck_fill(D, K)
    E3 = bisect_target(D, K, B_opt, B_eps)

    time_fn: Callable = phase_time_unsplittable if unsplittable else phase_time
    share_fn = optical_byte_share_unsplittable if unsplittable else None

    def eval_T(E: np.ndarray) -> float:
        if unsplittable:
            return phase_time_unsplittable(D, E, B_opt, B_eps)
        return phase_time(D, E, B_opt, B_eps)

    scores = {
        "WeightGreedy": (E1, eval_T(E1)),
        "BottleneckFill": (E2, eval_T(E2)),
        "BisectTarget": (E3, eval_T(E3)),
    }
    winner_name = min(scores, key=lambda n: scores[n][1])
    E_cand = scores[winner_name][0].copy()
    T_cand = scores[winner_name][1]

    repaired = False
    if use_repair:
        E_rep = local_swap_repair(D, E_cand, K, B_opt, B_eps)
        T_rep = eval_T(E_rep)
        if T_rep + 1e-15 < T_cand:
            E_cand = E_rep
            T_cand = T_rep
            repaired = True
            winner_name = winner_name + "+LocalSwapRepair"

    T_stay = eval_T(E_prev)
    T_move = T_cand + delta
    if T_move < T_stay - 1e-15:
        E_out, T_out, reconfigured = E_cand, T_move, True
        T_comm = T_cand
    else:
        E_out, T_out, reconfigured = E_prev, T_stay, False
        T_comm = T_stay

    if unsplittable:
        opt_b, tot_b = optical_byte_share_unsplittable(D, E_out)
    else:
        # Use communication time (without delta) for optical share accounting
        opt_b, tot_b = optical_byte_share(D, E_out, B_opt, T_comm)

    t1 = time.perf_counter()
    return {
        "E": E_out,
        "T_penalized": T_out,
        "T_comm": T_comm,
        "reconfigured": reconfigured,
        "winner": winner_name,
        "candidate_scores": {n: float(s[1]) for n, s in scores.items()},
        "repaired": repaired,
        "opt_bytes": opt_b,
        "total_bytes": tot_b,
        "latency_s": t1 - t0,
    }


def ghem_ac_with_repair(
    D: np.ndarray,
    E_prev: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    unsplittable: bool = False,
) -> dict[str, Any]:
    return ghem_ac(
        D, E_prev, K, B_opt, B_eps, delta,
        unsplittable=unsplittable, use_repair=True,
    )
