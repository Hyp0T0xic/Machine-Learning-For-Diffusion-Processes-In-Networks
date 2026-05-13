"""shared accent palette + canonical method→colour map + repo-root helper used by every plot in the project"""

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

# Canonical method → color mapping used by every plot in the project so that a
# given method always renders in the same hue. Groups:
#   • Random Forests (any flavour) — greens and teals
#       generic ``random_forest`` and BA-trained ``RF (IC-BA)`` share the same
#       green so synthetic-IC plots stay consistent with cross-dataset plots;
#       ``RF (IC-ER)`` is teal; ``rf_falsenews`` (validation-trained RF) is
#       a distinct sage-green.
#   • Centrality baselines — warm coral / pink / gold / rose
#   • Random guess — purple
#
# When a script trains/evaluates a single RF and stores it under the generic
# ``random_forest`` key, it should pick the colour for that specific RF (BA vs
# ER) at the call site — e.g. ``METHOD_COLORS["RF (IC-ER)"]`` for ER scripts.
METHOD_COLORS: dict[str, str] = {
    # RF family ─────────────────────────────────────────────
    "random_forest":  ACCENT_COLORS[5],   # generic RF → green (matches IC-BA)
    "RF (IC-BA)":     ACCENT_COLORS[5],   # green   #7FB382
    "RF (IC-ER)":     ACCENT_COLORS[2],   # teal    #4DB6AC
    "rf_falsenews":   ACCENT_COLORS[3],   # sage    #9CB067  (RF trained on validation set)

    # Centrality baselines ───────────────────────────────────
    "jordan":         ACCENT_COLORS[0],   # salmon  #D99685
    "closeness":      ACCENT_COLORS[1],   # pink    #E38EA0
    "degree":         ACCENT_COLORS[4],   # gold    #C0A064
    "betweenness":    ACCENT_COLORS[7],   # rose    #DA92B7

    # Random guess ───────────────────────────────────────────
    "random":         ACCENT_COLORS[8],   # purple  #C594D1
}

# Backwards-compatible alias for FalseNews validation scripts that imported
# ``FALSENEWS_METHOD_BAR_COLORS``. New code should use ``METHOD_COLORS``.
FALSENEWS_METHOD_BAR_COLORS: dict[str, str] = dict(METHOD_COLORS)


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


__all__ = [
    "ACCENT_COLORS",
    "METHOD_COLORS",
    "FALSENEWS_METHOD_BAR_COLORS",
    "repo_root",
    "method_bar_colors",
]
