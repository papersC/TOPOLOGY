"""Demand generators matching Section V of the manuscript."""

from __future__ import annotations

import numpy as np


def tensor_ring(N: int, volume_gb: float = 200.0) -> np.ndarray:
    """TP: unidirectional all-reduce ring; rack i sends volume_gb to (i+1) mod N."""
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        D[i, (i + 1) % N] = volume_gb
    return D


def expert_sparse(
    N: int,
    fanout: int = 4,
    vol_lo: float = 100.0,
    vol_hi: float = 400.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """EP: each rack targets `fanout` distinct destinations with U(vol_lo, vol_hi) Gb."""
    if rng is None:
        rng = np.random.default_rng(0)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        choices = list(range(N))
        choices.remove(i)
        dests = rng.choice(choices, size=min(fanout, N - 1), replace=False)
        for j in dests:
            D[i, int(j)] = float(rng.uniform(vol_lo, vol_hi))
    return D


def pipeline_adjacent(N: int, volume_gb: float = 800.0) -> np.ndarray:
    """PP: 800 Gb between adjacent pipeline stages (linear chain wrap optional).

    Stages form a ring of adjacent pairs so every rack participates:
    rack i sends volume_gb to rack (i+1) mod N (stage boundary traffic).
    """
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        D[i, (i + 1) % N] = volume_gb
    return D


def make_phase_sequence(
    N: int = 32,
    n_phases: int = 12,
    seed: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """Twelve phases cycling tensor / expert / pipeline (Section V).

    Order: TP, EP, PP repeated until n_phases. Expert draws use a dedicated
    RNG advanced per EP phase so seed=0 is reproducible.
    """
    rng = np.random.default_rng(seed)
    modes = ["TP", "EP", "PP"]
    phases: list[tuple[str, np.ndarray]] = []
    for m in range(n_phases):
        kind = modes[m % 3]
        if kind == "TP":
            D = tensor_ring(N)
        elif kind == "EP":
            D = expert_sparse(N, rng=rng)
        else:
            D = pipeline_adjacent(N)
        phases.append((kind, D))
    return phases


def perturb_demand(
    D: np.ndarray,
    noise_frac: float,
    rng: np.random.Generator,
    mode: str = "volume",
) -> np.ndarray:
    """Apply demand noise for uncertainty study.

    mode='volume': multiply each positive entry by Uniform(1-f, 1+f).
    mode='fanout': for EP-like matrices, randomly reassign one destination
                   per source with probability noise_frac (kept for API use).
    """
    Dp = D.copy()
    if mode == "volume":
        mask = D > 0
        factors = rng.uniform(1.0 - noise_frac, 1.0 + noise_frac, size=D.shape)
        Dp[mask] = D[mask] * factors[mask]
    elif mode == "fanout":
        N = D.shape[0]
        for i in range(N):
            js = np.where(D[i] > 0)[0]
            if len(js) == 0:
                continue
            if rng.random() < noise_frac:
                j_old = int(rng.choice(js))
                candidates = [j for j in range(N) if j != i and D[i, j] == 0]
                if not candidates:
                    continue
                j_new = int(rng.choice(candidates))
                Dp[i, j_new] = Dp[i, j_old]
                Dp[i, j_old] = 0.0
    else:
        raise ValueError(f"unknown noise mode: {mode}")
    return Dp


# ---------------------------------------------------------------------------
# Added in the Q1 revision: skewed (hotspot) expert-parallel workload used as
# a stress family in the scalability study.  Not used by the primary suite.
# ---------------------------------------------------------------------------

def expert_skewed(
    N: int,
    fanout: int = 8,
    zipf_s: float = 1.0,
    vol_lo: float = 100.0,
    vol_hi: float = 400.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """EP with Zipf-skewed expert popularity: destination j is chosen with
    probability proportional to 1/rank(j)^zipf_s (random rank permutation per
    phase), `fanout` distinct destinations per source, U(vol_lo, vol_hi) Gb."""
    if rng is None:
        rng = np.random.default_rng(0)
    D = np.zeros((N, N), dtype=float)
    ranks = rng.permutation(N) + 1
    w = 1.0 / ranks.astype(float) ** zipf_s
    for i in range(N):
        p = w.copy()
        p[i] = 0.0
        p /= p.sum()
        dests = rng.choice(N, size=min(fanout, N - 1), replace=False, p=p)
        for j in dests:
            D[i, int(j)] = float(rng.uniform(vol_lo, vol_hi))
    return D


def make_phase_sequence_skewed(
    N: int = 32,
    n_phases: int = 12,
    seed: int = 0,
    fanout: int = 8,
    zipf_s: float = 1.0,
) -> list[tuple[str, np.ndarray]]:
    """TP / skewed-EP / PP cycle (stress family)."""
    rng = np.random.default_rng(seed)
    modes = ["TP", "EP", "PP"]
    phases: list[tuple[str, np.ndarray]] = []
    for m in range(n_phases):
        kind = modes[m % 3]
        if kind == "TP":
            D = tensor_ring(N)
        elif kind == "EP":
            D = expert_skewed(N, fanout=fanout, zipf_s=zipf_s, rng=rng)
        else:
            D = pipeline_adjacent(N)
        phases.append((kind, D))
    return phases
