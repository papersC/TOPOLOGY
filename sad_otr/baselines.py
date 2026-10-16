"""Literature baselines and an ideal lower bound (added in the Q1 revision).

All baselines obey the same physical constraints as SAD-OTR:
  * each rack has K optical ports of rate B_opt (<= K circuits out and in),
  * at most one circuit between an ordered pair of racks at any instant,
  * a packet fallback with egress AND ingress capacity B_eps per rack,
  * every change of the circuit configuration costs delta seconds, during
    which the circuit layer carries nothing while the packet layer keeps
    forwarding.

Implemented schemes
-------------------
mwm_always   c-Through / Helios style: one max-weight degree-K configuration
             per phase (K rounds of Hungarian max-weight matching on D),
             adopted every phase (no amortization test).
solstice_k   Solstice (Liu et al., CoNEXT 2015): QuickStuff + BigSlice
             threshold slicing with durations set by the minimum stuffed entry,
             terminating once the unserved demand fits the packet switch.
             K-port generalisation: each slice adds up to K edge-disjoint
             perfect matchings found at the same threshold.
eclipse_k    Eclipse (Bojja Venkatakrishnan et al., SIGMETRICS 2016): greedy
             choice of (configuration, duration alpha) maximising
             served-bytes / (alpha + delta).  K-port configurations are built
             by K rounds of Hungarian max-weight matching.  Adapted from a
             fixed window to minimum completion time: the schedule prefix with
             the smallest completion time is kept (this can only favour
             Eclipse).
rotor        RotorNet-style schedule-oblivious rotation (direct one-hop
             forwarding): every ordered pair receives a time-averaged circuit
             rate K*B_opt*duty/(N-1).
ideal_lb     Non-blocking lower bound valid for every scheme above and for
             SAD-OTR: per-rack capacity B_eps + K*B_opt, per-pair capacity
             B_opt + B_eps, no reconfiguration cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .closed_form import optical_byte_share, phase_time


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def eps_time(R: np.ndarray, B_eps: float) -> float:
    """Packet-only time for residual R under egress+ingress caps."""
    if R.sum() <= 0:
        return 0.0
    return float(max(R.sum(axis=1).max(), R.sum(axis=0).max()) / B_eps)


def k_round_mwm(W: np.ndarray, K: int) -> np.ndarray:
    """Degree-<=K simple subgraph by K rounds of max-weight bipartite matching.

    Zero-weight edges are never kept.  Diagonal is excluded.
    """
    N = W.shape[0]
    Wc = W.astype(float).copy()
    np.fill_diagonal(Wc, 0.0)
    E = np.zeros((N, N), dtype=np.int8)
    for _ in range(K):
        if Wc.max() <= 0:
            break
        r, c = linear_sum_assignment(Wc, maximize=True)
        took = False
        for i, j in zip(r, c):
            if Wc[i, j] > 0:
                E[i, j] = 1
                Wc[i, j] = 0.0
                took = True
        if not took:
            break
    return E


def run_schedule(
    D: np.ndarray,
    schedule: list[tuple[np.ndarray, float]],
    B_opt: float,
    B_eps: float,
    delta: float,
    first_free: bool = False,
) -> dict[str, Any]:
    """Evaluate a multi-configuration circuit schedule with concurrent EPS.

    Each configuration k is held for dur_k after a delta reconfiguration.
    Circuits carry min(remaining, B_opt*dur_k) per edge; the residual is
    carried by the packet layer concurrently with circuit transmission.
    T = sum(delta charges) + max(sum dur_k, packet time of residual).
    """
    rem = D.astype(float).copy()
    C = 0.0          # circuit transmission time
    charges = 0.0    # reconfiguration time (packet layer counted only during
                     # transmission, as in the MILP / closed form)
    for k, (E, dur) in enumerate(schedule):
        C += dur
        charges += 0.0 if (first_free and k == 0) else delta
        sent = np.minimum(rem, (E > 0) * B_opt * dur)
        rem -= sent
    rem = np.maximum(rem, 0.0)
    T = charges + max(C, eps_time(rem, B_eps))
    total = float(D.sum())
    return {
        "T": T,
        "circuit_time": C,
        "n_configs": len(schedule),
        "opt_bytes": total - float(rem.sum()),
        "total_bytes": total,
    }


# ---------------------------------------------------------------------------
# Solstice (K-port generalisation)
# ---------------------------------------------------------------------------

def quick_stuff(D: np.ndarray) -> np.ndarray:
    """QuickStuff (Liu et al. 2015), off-diagonal only."""
    N = D.shape[0]
    S = D.astype(float).copy()
    np.fill_diagonal(S, 0.0)
    R = S.sum(axis=1)
    Cc = S.sum(axis=0)
    phi = max(R.max(), Cc.max())
    nz = [(i, j) for i in range(N) for j in range(N) if i != j and S[i, j] > 0]
    z = [(i, j) for i in range(N) for j in range(N) if i != j and S[i, j] == 0]
    for i, j in nz + z:
        add = phi - max(R[i], Cc[j])
        if add > 0:
            S[i, j] += add
            R[i] += add
            Cc[j] += add
    return S


def _perfect_matching(mask: np.ndarray) -> list[tuple[int, int]] | None:
    N = mask.shape[0]
    if not mask.any():
        return None
    m = maximum_bipartite_matching(csr_matrix(mask.astype(np.int8)), perm_type="column")
    if np.any(m < 0):
        return None
    return [(i, int(m[i])) for i in range(N)]


def solstice_k(
    D: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    max_configs: int = 10_000,
) -> dict[str, Any]:
    N = D.shape[0]
    Dp = quick_stuff(D)          # stuffed matrix D'
    Eres = D.astype(float).copy()  # real unserved demand E
    np.fill_diagonal(Eres, 0.0)
    T = 0.0
    schedule: list[tuple[np.ndarray, float]] = []
    mx = Dp.max()
    if mx <= 0:
        return {**run_schedule(D, [], B_opt, B_eps, delta), "schedule": []}
    r = 2.0 ** np.floor(np.log2(mx))
    if r >= mx:
        r /= 2.0
    r_floor = 1e-9 * mx
    while (max(Eres.sum(axis=1).max(), Eres.sum(axis=0).max()) > B_eps * T + 1e-12
           and len(schedule) < max_configs and r > r_floor):
        mask = Dp >= r
        np.fill_diagonal(mask, False)
        chosen: list[tuple[int, int]] = []
        for _ in range(K):
            pm = _perfect_matching(mask)
            if pm is None:
                break
            chosen.extend(pm)
            for i, j in pm:
                mask[i, j] = False
        if not chosen:
            r /= 2.0
            continue
        E = np.zeros((N, N), dtype=np.int8)
        for i, j in chosen:
            E[i, j] = 1
        dur = min(Dp[i, j] for i, j in chosen) / B_opt
        Dp[E > 0] -= B_opt * dur
        Dp[Dp < 1e-12 * mx] = 0.0
        Eres[E > 0] = np.maximum(0.0, Eres[E > 0] - B_opt * dur)
        T += dur          # transmission time (packet layer active)
        schedule.append((E, dur))
    out = run_schedule(D, schedule, B_opt, B_eps, delta)
    out["schedule"] = schedule
    return out


# ---------------------------------------------------------------------------
# Eclipse (K-port generalisation, min-completion adaptation)
# ---------------------------------------------------------------------------

def eclipse_k(
    D: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    max_alphas: int = 64,
    max_configs: int = 500,
) -> dict[str, Any]:
    N = D.shape[0]
    rem = D.astype(float).copy()
    np.fill_diagonal(rem, 0.0)
    schedule: list[tuple[np.ndarray, float]] = []
    C = 0.0
    best_T = eps_time(rem, B_eps)   # empty schedule (packet only)
    best_len = 0
    while rem.sum() > 0 and len(schedule) < max_configs:
        vals = np.unique(rem[rem > 0]) / B_opt
        if len(vals) > max_alphas:
            idx = np.unique(np.round(np.linspace(0, len(vals) - 1, max_alphas)).astype(int))
            vals = vals[idx]
        best = None
        for a in vals:
            W = np.minimum(rem, B_opt * a)
            E = k_round_mwm(W, K)
            gain = float((W * E).sum())
            score = gain / (a + delta)
            if best is None or score > best[0]:
                best = (score, E, float(a))
        if best is None or best[0] <= 0:
            break
        _, E, a = best
        rem = np.maximum(0.0, rem - (E > 0) * B_opt * a)
        C += a
        schedule.append((E, a))
        T_here = len(schedule) * delta + max(C, eps_time(rem, B_eps))
        if T_here < best_T:
            best_T, best_len = T_here, len(schedule)
        if len(schedule) * delta + C >= best_T:   # cannot improve further
            break
    schedule = schedule[:best_len]
    out = run_schedule(D, schedule, B_opt, B_eps, delta)
    out["schedule"] = schedule
    return out


# ---------------------------------------------------------------------------
# RotorNet-style rotation and ideal bound
# ---------------------------------------------------------------------------

def rotor_time(D: np.ndarray, K: int, B_opt: float, B_eps: float, duty: float = 0.9) -> dict[str, Any]:
    N = D.shape[0]
    E = np.ones((N, N), dtype=np.int8)
    np.fill_diagonal(E, 0)
    rate = K * B_opt * duty / (N - 1)
    T = phase_time(D, E, rate, B_eps)
    ob, tb = optical_byte_share(D, E, rate, T)
    return {"T": T, "opt_bytes": ob, "total_bytes": tb, "n_configs": 0}


def ideal_lower_bound(D: np.ndarray, K: int, B_opt: float, B_eps: float) -> float:
    cap = B_eps + K * B_opt
    row = D.sum(axis=1).max() / cap
    col = D.sum(axis=0).max() / cap
    pair = D.max() / (B_opt + B_eps)
    return float(max(row, col, pair))


def mwm_topology(D: np.ndarray, K: int) -> np.ndarray:
    return k_round_mwm(D, K)
