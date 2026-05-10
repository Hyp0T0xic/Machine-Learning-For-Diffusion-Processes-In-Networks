"""
Load FalseNews retweet cascades from raw CSV and convert to CascadeResult
objects for the ML pipeline. Strips edge direction + timestamps, truncates
via BFS to match simulated cascade sizes.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import networkx as nx
import pandas as pd

from src.data.cascade import CascadeResult


CSV_PATH = Path("data/FalseNews_Code_Data/data/raw_data_anon.csv")


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

        # root = tweet with parent_tid == -1
        roots = group[group["parent_tid"] == -1]
        if len(roots) == 0:
            skipped_no_root += 1
            continue

        root_row = roots.iloc[0]
        root_tid = int(root_row["tid"])

        # retweet tree: parent -> child edges
        tree = nx.DiGraph()
        all_tids = set(int(t) for t in group["tid"].values)
        tree.add_nodes_from(all_tids)

        for _, row in group.iterrows():
            parent = int(row["parent_tid"])
            child = int(row["tid"])
            # only add edge if parent is actually in this cascade
            if parent != -1 and parent in all_tids:
                tree.add_edge(parent, child)

        # only keep nodes reachable from root
        reachable = _bfs_truncate(tree, root_tid, target_size=len(all_tids))
        if len(reachable) < min_size:
            skipped_too_small += 1
            continue

        truncated = reachable[:target_size]
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
