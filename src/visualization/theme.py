"""
Paper / validation plotting palette and path helpers.

Use ``from src.visualization.theme import ACCENT_COLORS, repo_root`` in scripts
so colors stay aligned with validation figures and repo paths survive file moves.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

ACCENT_COLORS: tuple[str, ...] = (
    "#D99685",
    "#E38EA0",
    "#4DB6AC",
    "#9CB067",
    "#C0A064",
    "#7FB382",
    "#A3A1D8",
    "#DA92B7",
    "#C594D1",
    "#56B4BE",
)

_N_ACCENTS = len(ACCENT_COLORS)


def repo_root(start: Path | None = None) -> Path:
    """Directory containing ``pyproject.toml``, walking upward from ``start``.

    Parameters
    ----------
    start
        Path to start from (typically ``Path(__file__)`` from the caller).
        Defaults to this module's path so ``repo_root()`` works from here too.

    Raises
    ------
    FileNotFoundError
        If no ancestor contains ``pyproject.toml``.
    """
    current = (start if start is not None else Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError(
        f"No pyproject.toml found in parents of {current}"
    )


def method_bar_colors(
    method_order: Sequence[str],
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Map method keys to ``ACCENT_COLORS`` by stable index in ``method_order``.

    Index ``i`` in ``method_order`` maps to ``ACCENT_COLORS[i % 10]``, matching
    how validation bar charts cycle the palette.

    Parameters
    ----------
    method_order
        Canonical ordering (e.g. display / legend order).
    keys
        Subset of methods to include. If omitted, all entries in ``method_order``
        are returned.

    Raises
    ------
    KeyError
        If any requested key is missing from ``method_order``.
    """
    index = {name: i for i, name in enumerate(method_order)}
    want = list(method_order) if keys is None else list(keys)
    missing = [k for k in want if k not in index]
    if missing:
        raise KeyError(
            f"Keys not found in method_order: {missing!r}; "
            f"method_order={list(method_order)!r}"
        )
    return {k: ACCENT_COLORS[index[k] % _N_ACCENTS] for k in want}


__all__ = ["ACCENT_COLORS", "repo_root", "method_bar_colors"]
