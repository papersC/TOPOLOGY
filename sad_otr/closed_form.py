"""Closed-form phase duration T(D,E) with source AND destination EPS capacity."""

from __future__ import annotations

import numpy as np

# Absolute bisection tolerance used throughout experiments (seconds).
# Stated explicitly for Prop. complexity: eps = 1e-9 s; for N<=512 and
# T_hi ~ O(10) s, log(T_hi/eps) = O(log N) under the paper's accounting.
BISECT_EPS = 1e-9


def residual_matrix(D: np.ndarray, E: np.ndarray, B_opt: float, T: float) -> np.ndarray:
    """Packet residual after max optical offload on lightpaths in E."""
    opt_cap = (E > 0).astype(float) * B_opt * T
    return np.maximum(0.0, D - opt_cap)


def is_feasible(
    D: np.ndarray,
    E: np.ndarray,
    T: float,
    B_opt: float,
    B_eps: float,
) -> bool:
    """eq:feas with destination residuals: row and column packet caps."""
    if T < 0:
        return False
    R = residual_matrix(D, E, B_opt, T)
    row = R.sum(axis=1)
    col = R.sum(axis=0)
    cap = B_eps * T
    return bool(np.all(row <= cap + 1e-12) and np.all(col <= cap + 1e-12))


def _T_hi(D: np.ndarray, B_eps: float) -> float:
    """Packet-only upper bound under source and destination caps."""
    if B_eps <= 0:
        return float("inf")
    row = D.sum(axis=1).max()
    col = D.sum(axis=0).max()
    return float(max(row, col) / B_eps)


def phase_time(
    D: np.ndarray,
    E: np.ndarray,
    B_opt: float,
    B_eps: float,
    eps: float = BISECT_EPS,
) -> float:
    """Minimum T >= 0 satisfying eq:feas (source + destination EPS)."""
    if D.size == 0 or float(D.sum()) == 0.0:
        return 0.0
    hi = _T_hi(D, B_eps)
    if hi == 0.0:
        return 0.0
    # Also account for optical-only lower bound on edged demands.
    lo = 0.0
    if not is_feasible(D, E, hi, B_opt, B_eps):
        # Should not happen with B_eps > 0; expand defensively.
        hi *= 2.0
        while not is_feasible(D, E, hi, B_opt, B_eps):
            hi *= 2.0
    while hi - lo > eps:
        mid = 0.5 * (lo + hi)
        if is_feasible(D, E, mid, B_opt, B_eps):
            hi = mid
        else:
            lo = mid
    return float(hi)


def optical_byte_share(
    D: np.ndarray,
    E: np.ndarray,
    B_opt: float,
    T: float,
) -> tuple[float, float]:
    """Return (optical_bytes, total_bytes) at duration T with max optical offload."""
    total = float(D.sum())
    if total <= 0 or T <= 0:
        return 0.0, total
    opt = np.minimum(D, (E > 0).astype(float) * B_opt * T)
    # Only count optical on existing edges.
    opt = opt * (E > 0)
    return float(opt.sum()), total


def phase_time_unsplittable(
    D: np.ndarray,
    E: np.ndarray,
    B_opt: float,
    B_eps: float,
    eps: float = BISECT_EPS,
) -> float:
    """Unsplittable: if a lightpath exists, the whole demand uses optical
    (requiring T >= d_ij/B_opt); otherwise the whole demand uses packet.

    Packet residuals are exactly the demands without a lightpath; optical
    edges consume no packet capacity but force T >= max_{edged} d_ij/B_opt.
    """
    edged = E > 0
    # Optical lower bound
    if np.any(edged & (D > 0)):
        t_opt = float((D[edged].max()) / B_opt) if B_opt > 0 else float("inf")
    else:
        t_opt = 0.0
    # Packet-only residual matrix
    R = D.copy()
    R[edged] = 0.0
    t_eps = _T_hi(R, B_eps) if R.sum() > 0 else 0.0
    return max(t_opt, t_eps)


def optical_byte_share_unsplittable(
    D: np.ndarray,
    E: np.ndarray,
) -> tuple[float, float]:
    total = float(D.sum())
    opt = float(D[E > 0].sum())
    return opt, total
