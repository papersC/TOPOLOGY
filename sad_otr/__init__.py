"""SAD-OTR / GHEM-AC supplementary simulator."""

__version__ = "1.0.0"

from .demands import make_phase_sequence, tensor_ring, expert_sparse, pipeline_adjacent
from .closed_form import phase_time, is_feasible, optical_byte_share
from .matching import weight_greedy, bottleneck_fill, bisect_target, local_swap_repair
from .ghem_ac import ghem_ac, ghem_ac_with_repair
from .milp import solve_exact
from .evaluate import run_scheme, FabricParams

__all__ = [
    "make_phase_sequence",
    "tensor_ring",
    "expert_sparse",
    "pipeline_adjacent",
    "phase_time",
    "is_feasible",
    "optical_byte_share",
    "weight_greedy",
    "bottleneck_fill",
    "bisect_target",
    "local_swap_repair",
    "ghem_ac",
    "ghem_ac_with_repair",
    "solve_exact",
    "run_scheme",
    "FabricParams",
]
