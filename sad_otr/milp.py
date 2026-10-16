"""Exact MILP with PuLP/CBC: source AND destination packet capacity."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pulp


def solve_exact(
    D: np.ndarray,
    K: int,
    B_opt: float,
    B_eps: float,
    delta: float,
    E_prev: np.ndarray | None = None,
    unsplittable: bool = False,
    time_limit: float | None = 120.0,
    gap_rel: float = 0.0,
    threads: int = 1,
    msg: bool = False,
) -> dict[str, Any]:
    """Solve one-phase SAD-OTR MILP.

    CBC settings (documented for the manuscript):
      - PULP_CBC_CMD with timeLimit, gapRel, threads
      - msg=False for batch runs
      - Model-build time is recorded separately from solve wall time.
    """
    N = D.shape[0]
    if E_prev is None:
        E_prev = np.zeros((N, N), dtype=np.int8)

    t_build0 = time.perf_counter()
    prob = pulp.LpProblem("SAD_OTR", pulp.LpMinimize)

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j and D[i, j] > 1e-15]
    # Also include previous edges so reconfiguration constraints are complete
    prev_edges = [(i, j) for i in range(N) for j in range(N) if i != j and E_prev[i, j]]
    edge_universe = sorted(set(pairs) | set(prev_edges))

    x = pulp.LpVariable.dicts("x", edge_universe, cat="Binary")
    # Flows only on positive-demand pairs
    f_opt = pulp.LpVariable.dicts("fopt", pairs, lowBound=0)
    f_eps = pulp.LpVariable.dicts("feps", pairs, lowBound=0)
    T = pulp.LpVariable("T", lowBound=0)
    Delta = pulp.LpVariable("Delta", cat="Binary")

    prob += T + Delta * delta

    # Degree bounds (all possible directed pairs among N, but x only on universe;
    # for pairs not in universe x=0 implicitly)
    for i in range(N):
        outs = [x[i, j] for j in range(N) if (i, j) in x]
        if outs:
            prob += pulp.lpSum(outs) <= K
    for j in range(N):
        inns = [x[i, j] for i in range(N) if (i, j) in x]
        if inns:
            prob += pulp.lpSum(inns) <= K

    for i, j in pairs:
        d = float(D[i, j])
        prob += f_opt[i, j] + f_eps[i, j] == d
        if unsplittable:
            # All-or-nothing: f_opt = d * x, f_eps = d * (1-x)
            prob += f_opt[i, j] == d * x[i, j]
            prob += f_eps[i, j] == d * (1 - x[i, j])
        else:
            prob += f_opt[i, j] <= d * x[i, j]
        prob += f_opt[i, j] <= B_opt * T

    # Source packet capacity
    for i in range(N):
        terms = [f_eps[i, j] for j in range(N) if (i, j) in f_eps]
        if terms:
            prob += pulp.lpSum(terms) <= B_eps * T
    # Destination packet capacity
    for j in range(N):
        terms = [f_eps[i, j] for i in range(N) if (i, j) in f_eps]
        if terms:
            prob += pulp.lpSum(terms) <= B_eps * T

    # Reconfiguration indicators (edgewise)
    for i, j in edge_universe:
        prev = int(E_prev[i, j])
        prob += Delta >= x[i, j] - prev
        prob += Delta >= prev - x[i, j]

    t_build1 = time.perf_counter()

    solver = pulp.PULP_CBC_CMD(
        msg=msg,
        timeLimit=time_limit,
        gapRel=gap_rel,
        threads=threads,
    )
    t_sol0 = time.perf_counter()
    status = prob.solve(solver)
    t_sol1 = time.perf_counter()

    E = np.zeros((N, N), dtype=np.int8)
    for i, j in edge_universe:
        if pulp.value(x[i, j]) is not None and pulp.value(x[i, j]) > 0.5:
            E[i, j] = 1

    T_val = float(pulp.value(T) or 0.0)
    Delta_val = int(round(pulp.value(Delta) or 0.0))
    obj = T_val + Delta_val * delta

    opt_bytes = 0.0
    eps_bytes = 0.0
    for i, j in pairs:
        fo = float(pulp.value(f_opt[i, j]) or 0.0)
        fe = float(pulp.value(f_eps[i, j]) or 0.0)
        opt_bytes += fo
        eps_bytes += fe

    return {
        "status": pulp.LpStatus[status],
        # 1 = proven optimal, 2 = integer feasible only (e.g. time limit hit)
        "sol_status": int(getattr(prob, "sol_status", 0)),
        "E": E,
        "T": T_val,
        "Delta": Delta_val,
        "objective": obj,
        "opt_bytes": opt_bytes,
        "eps_bytes": eps_bytes,
        "total_bytes": opt_bytes + eps_bytes,
        "build_s": t_build1 - t_build0,
        "solve_s": t_sol1 - t_sol0,
        "wall_s": (t_build1 - t_build0) + (t_sol1 - t_sol0),
    }
