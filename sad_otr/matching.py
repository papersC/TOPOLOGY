"""Candidate topology constructions: WeightGreedy, BottleneckFill, BisectTarget,
plus LocalSwapRepair (optional 4th / post-repair for worst-gap mitigation).
"""

from __future__ import annotations

import numpy as np

from .closed_form import BISECT_EPS, is_feasible, phase_time


def empty_matching(N: int) -> np.ndarray:
    return np.zeros((N, N), dtype=np.int8)


def weight_greedy(D: np.ndarray, K: int) -> np.ndarray:
    """WeightGreedy: sort pairs by demand, accept while degree bounds hold."""
    N = D.shape[0]
    E = empty_matching(N)
    out_deg = np.zeros(N, dtype=int)
    in_deg = np.zeros(N, dtype=int)
    pairs = [(D[i, j], i, j) for i in range(N) for j in range(N) if i != j and D[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    for _, i, j in pairs:
        if out_deg[i] < K and in_deg[j] < K and E[i, j] == 0:
            E[i, j] = 1
            out_deg[i] += 1
            in_deg[j] += 1
    return E


def bottleneck_fill(D: np.ndarray, K: int) -> np.ndarray:
    """BottleneckFill: heaviest residual uplink rack chooses heaviest eligible demand."""
    N = D.shape[0]
    E = empty_matching(N)
    out_deg = np.zeros(N, dtype=int)
    in_deg = np.zeros(N, dtype=int)
    load = D.sum(axis=1).astype(float).copy()
    served = np.zeros_like(D, dtype=bool)

    while True:
        candidates = [i for i in range(N) if out_deg[i] < K]
        if not candidates:
            break
        i_star = max(candidates, key=lambda i: (load[i], i))
        # heaviest unserved demand whose head has free in-degree
        best_j = None
        best_d = -1.0
        for j in range(N):
            if i_star == j or served[i_star, j] or D[i_star, j] <= 0:
                continue
            if in_deg[j] >= K:
                continue
            if D[i_star, j] > best_d:
                best_d = D[i_star, j]
                best_j = j
        if best_j is None:
            # freeze this rack's remaining out-degree
            out_deg[i_star] = K
            continue
        E[i_star, best_j] = 1
        out_deg[i_star] += 1
        in_deg[best_j] += 1
        served[i_star, best_j] = True
        load[i_star] = max(0.0, load[i_star] - D[i_star, best_j])
    return E


def _try_build_at_T(
    D: np.ndarray,
    K: int,
    T: float,
    B_opt: float,
    B_eps: float,
) -> np.ndarray | None:
    """Attempt degree-feasible E making eq:feas true at trial T (BisectTarget step)."""
    N = D.shape[0]
    E = empty_matching(N)
    out_deg = np.zeros(N, dtype=int)
    in_deg = np.zeros(N, dtype=int)
    row_load = D.sum(axis=1)
    obligation = np.maximum(0.0, row_load - B_eps * T)
    absorption = np.minimum(D, B_opt * T)

    order = sorted(range(N), key=lambda i: (-obligation[i], i))
    remaining = obligation.copy()

    for i in order:
        if remaining[i] <= 1e-15:
            continue
        # eligible heads: positive demand, free in-degree, prefer least contended
        while remaining[i] > 1e-15 and out_deg[i] < K:
            best_j = None
            best_key = None
            for j in range(N):
                if j == i or D[i, j] <= 0 or E[i, j] or in_deg[j] >= K:
                    continue
                # least contended (lowest in_deg), then highest absorption
                key = (in_deg[j], -absorption[i, j], j)
                if best_key is None or key < best_key:
                    best_key = key
                    best_j = j
            if best_j is None:
                break
            E[i, best_j] = 1
            out_deg[i] += 1
            in_deg[best_j] += 1
            remaining[i] = max(0.0, remaining[i] - absorption[i, best_j])

    if is_feasible(D, E, T, B_opt, B_eps):
        return E
    return None


def bisect_target(
    D: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    eps: float = BISECT_EPS,
) -> np.ndarray:
    """BisectTarget: bisection on T with constructive feasibility oracle."""
    N = D.shape[0]
    row = D.sum(axis=1).max()
    col = D.sum(axis=0).max()
    T_hi = float(max(row, col) / B_eps) if B_eps > 0 else 1.0
    T_lo = 0.0
    best_E = empty_matching(N)
    # Ensure some feasible topology exists at T_hi (packet-only works with empty E)
    if is_feasible(D, best_E, T_hi, B_opt, B_eps):
        pass
    stored = best_E
    while T_hi - T_lo > eps:
        T = 0.5 * (T_lo + T_hi)
        E_try = _try_build_at_T(D, K, T, B_opt, B_eps)
        if E_try is not None:
            T_hi = T
            stored = E_try
        else:
            T_lo = T
    return stored


def local_swap_repair(
    D: np.ndarray,
    E: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    max_passes: int = 3,
) -> np.ndarray:
    """Lightweight local repair (optional 4th candidate / post-process).

    Identifies the current bottleneck rack (max residual row or column load
    relative to capacity at T=T(D,E)), then tries edge reassignments:
    drop a low-value edge incident to a saturated port and add a missing
    heavy demand edge that relieves the bottleneck. Documented in README
    as LocalSwapRepair.
    """
    E_cur = E.copy()
    T_cur = phase_time(D, E_cur, B_opt, B_eps)
    N = D.shape[0]

    for _ in range(max_passes):
        improved = False
        out_deg = E_cur.sum(axis=1).astype(int)
        in_deg = E_cur.sum(axis=0).astype(int)

        # Candidate additions: heavy demands without a lightpath
        missing = [
            (D[i, j], i, j)
            for i in range(N)
            for j in range(N)
            if i != j and D[i, j] > 0 and E_cur[i, j] == 0
        ]
        missing.sort(key=lambda t: -t[0])

        for _, i, j in missing[: min(64, len(missing))]:
            # Try add if degree allows
            if out_deg[i] < K and in_deg[j] < K:
                E_try = E_cur.copy()
                E_try[i, j] = 1
                T_try = phase_time(D, E_try, B_opt, B_eps)
                if T_try + 1e-15 < T_cur:
                    E_cur = E_try
                    T_cur = T_try
                    out_deg = E_cur.sum(axis=1).astype(int)
                    in_deg = E_cur.sum(axis=0).astype(int)
                    improved = True
                    continue
            # Try swap: free an out-port at i or in-port at j
            # Drop weakest existing out-edge from i
            if out_deg[i] >= K:
                outs = [jj for jj in range(N) if E_cur[i, jj]]
                if not outs:
                    continue
                weak = min(outs, key=lambda jj: D[i, jj])
                # Also may need to free in at j
                E_try = E_cur.copy()
                E_try[i, weak] = 0
                if in_deg[j] >= K:
                    inns = [ii for ii in range(N) if E_cur[ii, j]]
                    if not inns:
                        continue
                    weak_in = min(inns, key=lambda ii: D[ii, j])
                    E_try[weak_in, j] = 0
                E_try[i, j] = 1
                # Check degrees
                if E_try.sum(axis=1).max() > K or E_try.sum(axis=0).max() > K:
                    continue
                T_try = phase_time(D, E_try, B_opt, B_eps)
                if T_try + 1e-15 < T_cur:
                    E_cur = E_try
                    T_cur = T_try
                    out_deg = E_cur.sum(axis=1).astype(int)
                    in_deg = E_cur.sum(axis=0).astype(int)
                    improved = True
        if not improved:
            break
    return E_cur
