"""load falsenews retweet cascades from csv, bfs-truncate to target size, wrap as cascaderesult objects"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd

from src.data.cascade import CascadeResult


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _default_csv_path() -> Path:
    """Pick the first existing FalseNews raw CSV path."""
    candidates = [
        # current repo layout (folder checked into repo root)
        _REPO_ROOT / "FalseNews_Code_Data" / "FalseNews_Code_Data" / "data" / "raw_data_anon.csv",
        # legacy layout (nested under a top-level data/ folder)
        _REPO_ROOT / "data" / "FalseNews_Code_Data" / "data" / "raw_data_anon.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fall back to first candidate for a clear error message downstream
    return candidates[0]


CSV_PATH = _default_csv_path()


def _bfs_truncate(tree: nx.DiGraph, root: int, target_size: int) -> list[int]:
    """First N nodes from root in BFS order — mimics observing early spread."""
    visited: list[int] = []
    queue: deque[int] = deque([root])
    seen = {root}
    while queue and len(visited) < target_size:
        node = queue.popleft()
        visited.append(node)
        for child in tree.successors(node):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return visited


@dataclass
class _ParsedFalsenewsGroup:
    tree: nx.DiGraph
    root_tid: int
    reachable: list[int]
    truncated: list[int]
    root_row: pd.Series


def _parse_falsenews_group(
    group: pd.DataFrame,
    min_size: int,
    target_size: int,
) -> _ParsedFalsenewsGroup | Literal["no_root", "too_small"]:
    """Shared tree/BFS filter for loaders and export script."""
    roots = group[group["parent_tid"] == -1]
    if len(roots) == 0:
        return "no_root"

    root_row = roots.iloc[0]
    root_tid = int(root_row["tid"])

    tree = nx.DiGraph()
    all_tids = set(int(t) for t in group["tid"].values)
    tree.add_nodes_from(all_tids)

    for _, row in group.iterrows():
        parent = int(row["parent_tid"])
        child = int(row["tid"])
        if parent != -1 and parent in all_tids:
            tree.add_edge(parent, child)

    reachable = _bfs_truncate(tree, root_tid, target_size=len(all_tids))
    if len(reachable) < min_size:
        return "too_small"

    truncated = reachable[:target_size]
    return _ParsedFalsenewsGroup(
        tree=tree,
        root_tid=root_tid,
        reachable=reachable,
        truncated=truncated,
        root_row=root_row,
    )


def load_falsenews_cascades(
    csv_path: str | Path = CSV_PATH,
    target_size: int = 25,
    min_size: int = 25,
    veracity: str | None = None,
) -> tuple[list[CascadeResult], list[dict]]:
    """Load and filter FalseNews cascades.

    Returns (cascades, metadata) where metadata tracks veracity etc.
    for per-class analysis later. Could expand veracity filter to split
    TRUE vs FALSE vs MIXED runs.
    """
    csv_path = Path(csv_path)
    print(f"Loading FalseNews data from {csv_path} ...")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FalseNews CSV not found at: {csv_path}\n"
            f"Expected one of:\n"
            f"  - {_REPO_ROOT / 'FalseNews_Code_Data' / 'FalseNews_Code_Data' / 'data' / 'raw_data_anon.csv'}\n"
            f"  - {_REPO_ROOT / 'data' / 'FalseNews_Code_Data' / 'data' / 'raw_data_anon.csv'}\n"
            f"If you placed the dataset elsewhere, pass csv_path=... to load_falsenews_cascades()."
        )
    # The raw CSV is large; only load columns we need with stable dtypes.
    usecols = ["cascade_id", "tid", "parent_tid", "veracity", "rumor_category"]
    dtype = {
        "cascade_id": "int64",
        "tid": "int64",
        "parent_tid": "int64",
        "veracity": "string",
        "rumor_category": "string",
    }
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, low_memory=False)

    if veracity is not None:
        df = df[df["veracity"] == veracity.upper()]

    grouped = df.groupby("cascade_id")
    total_groups = len(grouped)

    cascades: list[CascadeResult] = []
    metadata: list[dict] = []

    skipped_no_root = 0
    skipped_too_small = 0

    for idx, (cascade_id, group) in enumerate(grouped):
        if idx % 5000 == 0 and idx > 0:
            print(f"  Processed {idx}/{total_groups} cascade groups ...")

        parsed = _parse_falsenews_group(group, min_size, target_size)
        if parsed == "no_root":
            skipped_no_root += 1
            continue
        if parsed == "too_small":
            skipped_too_small += 1
            continue

        tree = parsed.tree
        root_tid = parsed.root_tid
        reachable = parsed.reachable
        truncated = parsed.truncated
        root_row = parsed.root_row
        truncated_set = set(truncated)

        cascade_edges = [
            (u, v) for u, v in tree.edges()
            if u in truncated_set and v in truncated_set
        ]

        # use BFS depth as infection time (analogous to IC timesteps)
        sub = tree.subgraph(truncated_set)
        depths = nx.single_source_shortest_path_length(sub, root_tid)
        infection_times = {node: depths.get(node, 0) for node in truncated}

        cascades.append(CascadeResult(
            source=root_tid,
            model_name="RealData",
            params={},
            infection_times=infection_times,
            cascade_edges=cascade_edges,
            network_name="FalseNews",
        ))
        metadata.append({
            "cascade_id": int(cascade_id),
            "veracity": str(root_row["veracity"]),
            "rumor_category": str(root_row.get("rumor_category", "")),
            "original_size": len(reachable),
        })

    print(f"\nLoaded {len(cascades)} cascades "
          f"(target_size={target_size}, min_size={min_size})")
    print(f"  Skipped: {skipped_no_root} no root, "
          f"{skipped_too_small} too small (<{min_size} reachable nodes)")
    if metadata:
        veracities = pd.Series([m["veracity"] for m in metadata]).value_counts()
        print(f"  Veracity breakdown: {dict(veracities)}")

    return cascades, metadata
