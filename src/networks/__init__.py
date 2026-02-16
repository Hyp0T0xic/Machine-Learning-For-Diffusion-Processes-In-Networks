"""Network generation package for diffusion cascade source detection."""

from .generator import (
    generate_er_network,
    generate_ba_network,
    generate_complete_graph,
    generate_all_networks,
    compute_network_stats,
    save_networks,
)

__all__ = [
    "generate_er_network",
    "generate_ba_network",
    "generate_complete_graph",
    "generate_all_networks",
    "compute_network_stats",
    "save_networks",
]
