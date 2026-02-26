"""
src.utils
=========
Shared utilities used across the pipeline.

Functions
---------
set_seed    : Set random seeds for Python, NumPy (and optionally PyTorch).
load_config : Load a YAML config file into a dict.
pretty_dict : Format a nested dict as a human-readable string for logging.
"""
from __future__ import annotations

import random
import pprint
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set Python and NumPy random seeds for reproducibility.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pretty_dict(d: dict, indent: int = 2) -> str:
    """Return a human-readable string representation of a dictionary."""
    return pprint.pformat(d, indent=indent)
