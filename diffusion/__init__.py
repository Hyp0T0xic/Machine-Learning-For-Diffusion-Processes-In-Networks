"""Diffusion simulation package for cascade source detection."""

from .models import (
    CascadeResult,
    IndependentCascade,
    SIModel,
    SIRModel,
    create_model,
    r0_to_params,
)
from .simulator import (
    select_sources,
    run_experiment,
    compute_cascade_stats,
    save_cascades,
    load_cascades,
)
from .cascade_viz import plot_cascade_tree, plot_comparison_grid

__all__ = [
    "CascadeResult",
    "IndependentCascade",
    "SIModel",
    "SIRModel",
    "create_model",
    "r0_to_params",
    "select_sources",
    "run_experiment",
    "compute_cascade_stats",
    "save_cascades",
    "load_cascades",
    "plot_cascade_tree",
    "plot_comparison_grid",
]
