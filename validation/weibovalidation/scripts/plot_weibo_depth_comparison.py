#!/usr/bin/env python
"""Generate a Weibo cascade comparison figure with specific depths: 2 and 8."""
from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.cascade import CascadeResult
from src.visualization.cascades import plot_cascade_tree

WEIBO_DIR = _REPO_ROOT / "rumdect" / "Weibo"
MIN_SIZE = 25

def _load_full_cascade(json_path: Path) -> CascadeResult | None:
    """Build a CascadeResult from a Weibo JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception:
        return None

    if not posts: return None

    mid_to_int = {}
    counter = 0
    def get_id(mid: str) -> int:
        nonlocal counter
        if mid not in mid_to_int:
            mid_to_int[mid] = counter
            counter += 1
        return mid_to_int[mid]

    source_mid = None
    for p in posts:
        if p.get("parent") is None:
            source_mid = str(p.get("mid") or p.get("id"))
            break
    
    if source_mid is None: return None
    
    source_int = get_id(source_mid)
    infection_times = {}
    cascade_edges = []
    
    # Sort by time for relative infection hours
    sorted_posts = sorted(posts, key=lambda p: p.get("t", 0))
    t0 = sorted_posts[0].get("t", 0) if sorted_posts else 0

    for post in sorted_posts:
        mid = str(post.get("mid") or post.get("id"))
        node_id = get_id(mid)
        parent_mid = post.get("parent")
        rel_time = (post.get("t", 0) - t0) // 3600
        infection_times[node_id] = rel_time
        
        if parent_mid is not None:
            parent_mid_str = str(parent_mid)
            if parent_mid_str in mid_to_int:
                cascade_edges.append((mid_to_int[parent_mid_str], node_id))

    return CascadeResult(
        source=source_int,
        model_name="Weibo_Repost",
        params={},
        infection_times=infection_times,
        cascade_edges=cascade_edges,
        network_name="Weibo"
    )

def main():
    print(f"Scanning Weibo cascades in {WEIBO_DIR}...")
    json_files = sorted(WEIBO_DIR.glob("*.json"))
    
    target_depths = [2, 8]
    selected = {d: None for d in target_depths}
    
    for jf in json_files:
        if all(selected.values()): break
        
        c = _load_full_cascade(jf)
        if c and c.size >= MIN_SIZE:
            d = c.depth
            if d in selected and selected[d] is None:
                selected[d] = c
                print(f"  Found cascade with depth {d} (size {c.size})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("white")
    
    import matplotlib.patches as mpatches

    for i, d in enumerate(target_depths):
        ax = axes[i]
        cascade = selected[d]
        if cascade:
            plot_cascade_tree(cascade, ax=ax)
            ax.set_title(
                f"Depth = {d}  |  {cascade.size} nodes",
                fontsize=14, fontweight="bold"
            )
            # Remove the per-axis legend that plot_cascade_tree adds
            legend = ax.get_legend()
            if legend:
                legend.remove()
        else:
            ax.text(0.5, 0.5, f"No cascade found with depth {d}", ha='center')

    # Single shared legend placed below the subplots
    cmap = plt.cm.YlOrRd
    legend_handles = [
        mpatches.Patch(color="red", label="Source (Patient Zero)"),
        mpatches.Patch(color=cmap(0.3), label="Early infection"),
        mpatches.Patch(color=cmap(0.9), label="Late infection"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Weibo Repost Cascade Examples", fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = _REPO_ROOT / "validation/weibovalidation/figures/cascades/weibo_depth_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved specific depth comparison plot -> {out_path}")

if __name__ == "__main__":
    main()
