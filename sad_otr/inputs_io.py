"""Save/load phase sequences (list of (kind, N x N demand matrix in Gb)) as .npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_phases(path, phases, **meta) -> None:
    arrs = {f"D{m:02d}": np.asarray(D, dtype=np.float64) for m, (_, D) in enumerate(phases)}
    arrs["kinds"] = np.array([k for k, _ in phases])
    for k, v in meta.items():
        arrs[f"meta_{k}"] = np.array(v)
    np.savez_compressed(path, **arrs)


def load_phases(path) -> list[tuple[str, np.ndarray]]:
    z = np.load(Path(path), allow_pickle=False)
    kinds = [str(k) for k in z["kinds"]]
    return [(kinds[m], z[f"D{m:02d}"]) for m in range(len(kinds))]
