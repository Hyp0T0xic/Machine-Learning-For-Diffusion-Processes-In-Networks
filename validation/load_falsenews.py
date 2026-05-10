"""
validation.load_falsenews
=========================
Load FalseNews retweet cascades from the raw CSV and convert them
into CascadeResult objects compatible with the existing ML pipeline.

Each cascade is a retweet tree.  We strip edge direction and timestamps,
truncate to a target size via BFS from the true root, and wrap the
result in the same CascadeResult dataclass used by the simulated data.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import networkx as nx
import pandas as pd

from src.data.cascade import CascadeResult


CSV_PATH = Path("data/FalseNews_Code_Data/data/raw_data_anon.csv")


def _bfs_truncate(tree: nx.DiGraph, root: int, target_size: int) -> list[int]:
    """Return the first *target_size* nodes reachable from *root* in BFS order."""
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


def load_falsenews_cascades(
    csv_path: str | Path = CSV_PATH,
    target_size: int = 25,
    min_size: int = 25,
    veracity: str | None = None,
) -> tuple[list[CascadeResult], list[dict]]:
    """Load FalseNews retweet cascades and convert to CascadeResult format.

    Parameters
    ----------
    csv_path : path to raw_data_anon.csv
    target_size : truncate cascades to this many nodes via BFS from root
    min_size : minimum reachable-from-root size to include
    veracity : filter to ``"TRUE"``, ``"FALSE"``, or ``"MIXED"`` (None = all)

    Returns
    -------
    cascades : list[CascadeResult]
    metadata : list[dict]
        Per-cascade info: cascade_id, veracity, rumor_category, original_size.
    """
    print(f"Loading FalseNews data from {csv_path} ...")
    df = pd.read_csv(csv_path)

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

        roots = group[group["parent_tid"] == -1]
        if len(roots) == 0:
            skipped_no_root += 1
            continue

        root_row = roots.iloc[0]
        root_tid = int(root_row["tid"])

        # Build directed tree: parent -> child
        tree = nx.DiGraph()
        all_tids = set(int(t) for t in group["tid"].values)
        tree.add_nodes_from(all_tids)

        for _, row in group.iterrows():
            parent = int(row["parent_tid"])
            child = int(row["tid"])
            if parent != -1 and parent in all_tids:
                tree.add_edge(parent, child)

        # BFS from root — only keep reachable nodes
        reachable = _bfs_truncate(tree, root_tid, target_size=len(all_tids))
        if len(reachable) < min_size:
            skipped_too_small += 1
            continue

        # Truncate to target_size
        truncated = reachable[:target_size]
        truncated_set = set(truncated)

        cascade_edges = [
            (u, v) for u, v in tree.edges()
            if u in truncated_set and v in truncated_set
        ]

        # Infection times = BFS depth from root (analogous to IC timesteps)
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
