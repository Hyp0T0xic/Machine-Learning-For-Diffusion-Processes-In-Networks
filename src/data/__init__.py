"""
src.data — Graph generation, epidemic simulation, and cascade I/O.

Modules
-------
networks  : Generate and persist ER, BA, and Complete contact graphs.
cascade   : CascadeResult dataclass + IC, SI, SIR epidemic model classes.
simulate  : Experiment runner — source selection, batch simulation, JSON I/O.
"""
from src.data.cascade import CascadeResult, IndependentCascade, SIModel, SIRModel, create_model, r0_to_params
from src.data.networks import generate_all_networks, compute_network_stats, save_networks
from src.data.simulate import select_sources, run_experiment, compute_cascade_stats, save_cascades, load_cascades

__all__ = [
    "CascadeResult", "IndependentCascade", "SIModel", "SIRModel", "create_model", "r0_to_params",
    "generate_all_networks", "compute_network_stats", "save_networks",
    "select_sources", "run_experiment", "compute_cascade_stats", "save_cascades", "load_cascades",
]
