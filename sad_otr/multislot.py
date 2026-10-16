"""Multi-configuration (intra-phase time-multiplexed) schedules.

* ``retime_lp``: given an ordered list of circuit configurations and their
  reconfiguration charges, choose durations and the optical/packet split by a
  linear program that minimises the phase completion time.  With a single
  configuration it reduces exactly to the closed form T(D,E) (Prop. 1).
  Applying it to a baseline's configuration sequence can only help that
  baseline.
* ``sad_otr_tm``: slice-level extension of the amortization rule.  The GHEM-AC
  topology is the first slice (free when it equals the incumbent); further
  slices are generated greedily (Eclipse-style gain/(alpha+delta)) on the
  residual, and a slice prefix is adopted only if its LP-retimed completion
  time, including delta per added slice, is lower than the shorter prefix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .baselines import eps_time, k_round_mwm
from .ghem_ac import ghem_ac


def retime_lp(
    D: np.ndarray,
    configs: list[np.ndarray],
    charges: list[float],
    B_opt: float,
    B_eps: float,
) -> dict[str, Any]:
    N = D.shape[0]
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j and D[i, j] > 0]
    pidx = {p: n for n, p in enumerate(pairs)}
    m = len(configs)
    # variable layout: [tau, t_1..t_m, f_(e,k) ...]; tau = transmission time.
    # Convention identical to the MILP / closed form: the packet layer is
    # counted only during transmission (tau), and the phase lasts
    # T = tau + sum(charges).
    fvars: list[tuple[int, int]] = []  # (pair index, k)
    for k, E in enumerate(configs):
        for (i, j), n in pidx.items():
            if E[i, j]:
                fvars.append((n, k))
    nv = 1 + m + len(fvars)
    c = np.zeros(nv)
    c[0] = 1.0
    rows, cols, vals, b = [], [], [], []
    r = 0
    # sum_k t_k - tau <= 0
    rows += [r]; cols += [0]; vals += [-1.0]
    for k in range(m):
        rows += [r]; cols += [1 + k]; vals += [1.0]
    b.append(0.0); r += 1
    # f_(e,k) - B_opt t_k <= 0
    for v, (n, k) in enumerate(fvars):
        rows += [r, r]; cols += [1 + m + v, 1 + k]; vals += [1.0, -B_opt]
        b.append(0.0); r += 1
    # sum_k f_(e,k) <= d_e
    by_pair: dict[int, list[int]] = {}
    for v, (n, k) in enumerate(fvars):
        by_pair.setdefault(n, []).append(v)
    for n, vs in by_pair.items():
        for v in vs:
            rows += [r]; cols += [1 + m + v]; vals += [1.0]
        i, j = pairs[n]
        b.append(float(D[i, j])); r += 1
    # packet egress: sum_j d_ij - sum_{j,k} f_ijk <= B_eps T
    for side in (0, 1):
        for a in range(N):
            tot = 0.0
            rows += [r]; cols += [0]; vals += [-B_eps]
            for n, (i, j) in enumerate(pairs):
                if (i if side == 0 else j) == a:
                    tot += D[i, j]
                    for v in by_pair.get(n, []):
                        rows += [r]; cols += [1 + m + v]; vals += [-1.0]
            b.append(-tot); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    res = linprog(c, A_ub=A, b_ub=np.array(b), bounds=[(0, None)] * nv, method="highs")
    if res.status != 0:
        raise RuntimeError(f"retime LP failed: {res.message}")
    x = res.x
    opt_bytes = float(x[1 + m:].sum()) if fvars else 0.0
    return {
        "T": float(x[0]) + float(sum(charges)),
        "tau": float(x[0]),
        "durations": [float(t) for t in x[1:1 + m]],
        "opt_bytes": opt_bytes,
        "total_bytes": float(D.sum()),
        "n_configs": m,
        "n_charged": int(sum(1 for ch in charges if ch > 0)),
    }


def iter_greedy_slices(
    rem: np.ndarray,
    K: int,
    B_opt: float,
    delta: float,
    n_max: int,
    max_alphas: int = 64,
):
    """Eclipse-style greedy slices on a residual matrix (lazy generator)."""
    rem = rem.astype(float).copy()
    for _ in range(n_max):
        if rem.sum() <= 0:
            break
        vals = np.unique(rem[rem > 0]) / B_opt
        if len(vals) > max_alphas:
            idx = np.unique(np.round(np.linspace(0, len(vals) - 1, max_alphas)).astype(int))
            vals = vals[idx]
        best = None
        for a in vals:
            W = np.minimum(rem, B_opt * a)
            E = k_round_mwm(W, K)
            score = float((W * E).sum()) / (a + delta)
            if best is None or score > best[0]:
                best = (score, E, float(a))
        if best is None or best[0] <= 0:
            break
        _, E, a = best
        rem = np.maximum(0.0, rem - (E > 0) * B_opt * a)
        yield (E, a)


def greedy_slices(rem, K, B_opt, delta, n_max, max_alphas=64):
    return list(iter_greedy_slices(rem, K, B_opt, delta, n_max, max_alphas))


def best_retimed_prefix(
    D: np.ndarray,
    configs: list[np.ndarray],
    charges: list[float],
    B_opt: float,
    B_eps: float,
    min_len: int = 1,
) -> dict[str, Any]:
    """LP-retime every prefix (length >= min_len) and keep the fastest."""
    best = None
    if min_len == 0:
        best = {"T": eps_time(D, B_eps), "durations": [], "opt_bytes": 0.0,
                "total_bytes": float(D.sum()), "n_configs": 0, "n_charged": 0, "len": 0}
    for L in range(max(1, min_len), len(configs) + 1):
        r = retime_lp(D, configs[:L], charges[:L], B_opt, B_eps)
        r["len"] = L
        if best is None or r["T"] < best["T"] - 1e-12:
            best = r
        # lower bound on any longer prefix: sum of its charges
        if sum(charges[:L + 1]) >= best["T"]:
            break
    return best


def sad_otr_tm(
    D: np.ndarray,
    E_prev: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    n_slices: int | None = None,
) -> dict[str, Any]:
    N = D.shape[0]
    g = ghem_ac(D, E_prev, K, B_opt, B_eps, delta)
    E0 = g["E"]
    c0 = delta if g["reconfigured"] else 0.0
    # residual after E0 held for its single-slice duration
    rem = np.maximum(0.0, D - (E0 > 0) * B_opt * g["T_comm"] * 0.0)
    # alpha0 chosen by the same gain/(alpha+delta) score
    vals = np.unique(D[(E0 > 0) & (D > 0)]) / B_opt if np.any((E0 > 0) & (D > 0)) else np.array([])
    if len(vals):
        scores = [float(np.minimum(D, B_opt * a)[E0 > 0].sum()) / (a + delta) for a in vals]
        a0 = float(vals[int(np.argmax(scores))])
        rem = np.maximum(0.0, D - (E0 > 0) * B_opt * a0)
    if n_slices is None:
        n_slices = 4 * max(1, int(np.ceil(N / K)))
    # Lazy slice-level amortization: a slice is appended only while the
    # reconfiguration time already committed is below the best LP-retimed
    # completion time found so far (identical result to evaluating every
    # prefix of the full greedy sequence, at lower cost).
    configs = [E0]
    charges = [c0]
    best = retime_lp(D, configs, charges, B_opt, B_eps)
    best["len"] = 1
    for E, _ in iter_greedy_slices(rem, K, B_opt, delta, n_slices):
        if sum(charges) + delta >= best["T"]:
            break
        configs.append(E)
        charges.append(delta)
        r = retime_lp(D, configs, charges, B_opt, B_eps)
        r["len"] = len(configs)
        if r["T"] < best["T"] - 1e-12:
            best = r
    L = best["len"]
    return {
        "T": best["T"],
        "n_slices": L,
        "n_charged": best["n_charged"],
        "opt_bytes": best["opt_bytes"],
        "total_bytes": best["total_bytes"],
        "E_last": configs[L - 1],
        "configs": configs[:L],
        "charges": charges[:L],
        "durations": best["durations"],
        "ghem_T": g["T_penalized"],
    }


def sad_otr_tm_variant(
    D: np.ndarray,
    E_prev: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    n_slices: int | None = None,
    max_alphas: int = 64,
    lazy_stop: bool = True,
    adopt_all: bool = False,
    first_slice: str = "ghem",
    time_budget_s: float | None = None,
) -> dict[str, Any]:
    """Parameterised SAD-OTR-TM for the ablation study (final-check revision).

    n_slices      slice cap L_max (default 4*ceil(N/K), as in sad_otr_tm)
    max_alphas    size of the candidate-duration set per greedy slice
    lazy_stop     stop generating once committed charges reach the incumbent
                  (False: generate up to the cap and keep the best prefix)
    adopt_all     amortization OFF: adopt every generated slice (no best-prefix
                  selection; slices generated up to the cap or until demand is
                  exhausted)
    first_slice   'ghem'   : GHEM-AC topology, free if it equals E_prev (default)
                  'ghem_charged': GHEM-AC topology, always charged delta
                  'greedy' : no GHEM-AC slice; all slices greedy and charged
    time_budget_s stop adding slices once this wall-clock budget is used
                  (the best prefix found so far is returned)
    With defaults the result equals sad_otr_tm exactly.
    """
    import time as _time
    t_start = _time.perf_counter()
    N = D.shape[0]
    if n_slices is None:
        n_slices = 4 * max(1, int(np.ceil(N / K)))
    configs: list[np.ndarray] = []
    charges: list[float] = []
    if first_slice in ("ghem", "ghem_charged"):
        g = ghem_ac(D, E_prev, K, B_opt, B_eps, delta)
        E0 = g["E"]
        if first_slice == "ghem":
            c0 = delta if g["reconfigured"] else 0.0
        else:
            c0 = delta
        rem = D.astype(float).copy()
        vals = np.unique(D[(E0 > 0) & (D > 0)]) / B_opt if np.any((E0 > 0) & (D > 0)) else np.array([])
        if len(vals):
            scores = [float(np.minimum(D, B_opt * a)[E0 > 0].sum()) / (a + delta) for a in vals]
            a0 = float(vals[int(np.argmax(scores))])
            rem = np.maximum(0.0, D - (E0 > 0) * B_opt * a0)
        configs, charges = [E0], [c0]
        best = retime_lp(D, configs, charges, B_opt, B_eps)
        best["len"] = 1
        budget_left = n_slices - 1
    elif first_slice == "greedy":
        rem = D.astype(float).copy()
        best = {"T": eps_time(D, B_eps), "durations": [], "opt_bytes": 0.0,
                "total_bytes": float(D.sum()), "n_configs": 0, "n_charged": 0, "len": 0}
        budget_left = n_slices
    else:
        raise ValueError(first_slice)
    last = best
    for E, _ in iter_greedy_slices(rem, K, B_opt, delta, budget_left, max_alphas):
        if time_budget_s is not None and _time.perf_counter() - t_start > time_budget_s:
            break
        if lazy_stop and not adopt_all and sum(charges) + delta >= best["T"]:
            break
        configs.append(E)
        charges.append(delta)
        r = retime_lp(D, configs, charges, B_opt, B_eps)
        r["len"] = len(configs)
        last = r
        if r["T"] < best["T"] - 1e-12:
            best = r
    if adopt_all:
        best = last
    L = best["len"]
    return {
        "T": best["T"], "n_slices": L, "n_charged": best["n_charged"],
        "opt_bytes": best["opt_bytes"], "total_bytes": float(D.sum()),
        "E_last": configs[L - 1] if L > 0 else E_prev,
        "n_generated": len(configs),
    }
