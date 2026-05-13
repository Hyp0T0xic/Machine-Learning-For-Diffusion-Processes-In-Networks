#!/usr/bin/env python
"""filter the falsenews csv down to cascades with a root and >= min_size reachable nodes.
two row modes: 'reachable' keeps every row in the rooted subtree, 'subgraph' keeps only the bfs-first target_size."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from validation.load_falsenews import (  # noqa: E402
    CSV_PATH,
    _parse_falsenews_group,
)


def _build_allowed_pairs(
    df: pd.DataFrame,
    min_size: int,
    target_size: int,
    row_mode: str,
    veracity: str | None,
) -> tuple[pd.DataFrame, dict]:
    """Pass 1: same filter as loader; return (allowed[cascade_id,tid], stats)."""
    if veracity is not None:
        df = df[df["veracity"] == veracity.upper()].copy()

    grouped = df.groupby("cascade_id")
    total_groups = len(grouped)

    skipped_no_root = 0
    skipped_too_small = 0
    kept_cascade_ids: list[int] = []
    pair_rows: list[dict] = []

    t0 = time.perf_counter()
    for idx, (cascade_id, group) in enumerate(grouped):
        if idx > 0 and idx % 5000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Pass 1: {idx}/{total_groups} groups ({elapsed:.1f}s) ...")

        parsed = _parse_falsenews_group(group, min_size, target_size)
        if parsed == "no_root":
            skipped_no_root += 1
            continue
        if parsed == "too_small":
            skipped_too_small += 1
            continue

        kept_cascade_ids.append(int(cascade_id))
        tids = parsed.truncated if row_mode == "subgraph" else parsed.reachable
        for t in tids:
            pair_rows.append({"cascade_id": int(cascade_id), "tid": int(t)})

    allowed = pd.DataFrame(pair_rows)
    stats = {
        "total_groups": total_groups,
        "kept_cascades": len(kept_cascade_ids),
        "skipped_no_root": skipped_no_root,
        "skipped_too_small": skipped_too_small,
        "row_mode": row_mode,
        "min_size": min_size,
        "target_size": target_size,
        "veracity_filter": veracity,
        "allowed_row_pairs": int(len(allowed)),
    }
    return allowed, stats


def _pass2_streaming_write(
    input_path: Path,
    output_path: Path,
    allowed: pd.DataFrame,
    chunksize: int,
) -> int:
    """Inner-join each chunk to allowed (cascade_id, tid); write CSV."""
    if allowed.empty:
        raise ValueError("No rows passed the filter; refusing to write empty output.")

    allowed = allowed.drop_duplicates()
    out_rows = 0
    first_chunk = True
    t0 = time.perf_counter()

    reader = pd.read_csv(
        input_path,
        chunksize=chunksize,
        dtype={"cascade_id": "int64", "tid": "int64"},
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        merged = chunk.merge(allowed, on=["cascade_id", "tid"], how="inner")
        out_rows += len(merged)
        merged.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Pass 2: {i + 1} chunks read, {out_rows} rows written ({elapsed:.1f}s) ...")

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=CSV_PATH,
        help="Source raw_data_anon.csv (default: discovered repo path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: alongside input, name encodes options)",
    )
    parser.add_argument("--min-size", type=int, default=25)
    parser.add_argument("--target-size", type=int, default=25)
    parser.add_argument(
        "--row-mode",
        choices=("reachable", "subgraph"),
        default="reachable",
        help="reachable = full rooted subtree; subgraph = BFS-first target_size nodes",
    )
    parser.add_argument(
        "--veracity",
        default=None,
        help="If set, only cascades whose rows match this veracity (TRUE/FALSE/MIXED)",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional path to write filter statistics JSON",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.output is None:
        suffix = f"min{args.min_size}_{args.row_mode}.csv"
        output_path = input_path.with_name(f"{input_path.stem}_filtered_{suffix}")
    else:
        output_path = args.output.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Filter: min_size={args.min_size}, target_size={args.target_size}, "
        f"row_mode={args.row_mode}"
    )

    usecols = ["cascade_id", "tid", "parent_tid", "veracity", "rumor_category"]
    dtype = {
        "cascade_id": "int64",
        "tid": "int64",
        "parent_tid": "int64",
        "veracity": "string",
        "rumor_category": "string",
    }
    print("Pass 1: reading minimal columns and building keep list ...")
    df = pd.read_csv(input_path, usecols=usecols, dtype=dtype, low_memory=False)
    allowed, stats = _build_allowed_pairs(
        df,
        min_size=args.min_size,
        target_size=args.target_size,
        row_mode=args.row_mode,
        veracity=args.veracity,
    )
    del df
    print(f"  Kept {stats['kept_cascades']} cascades, "
          f"{stats['allowed_row_pairs']} (cascade_id, tid) pairs for merge.")
    print(f"  Skipped: {stats['skipped_no_root']} no root, "
          f"{stats['skipped_too_small']} too small.")

    print("Pass 2: streaming full CSV with column join ...")
    written = _pass2_streaming_write(
        input_path, output_path, allowed, chunksize=args.chunksize
    )
    stats["output_rows"] = written
    stats["output_path"] = str(output_path)

    print(f"\nDone. Wrote {written} rows -> {output_path}")
    print("Original file unchanged. Point loaders at the new path, e.g.:")
    print(f"  load_falsenews_cascades(csv_path=r'{output_path}')")

    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Stats -> {args.stats_json}")


if __name__ == "__main__":
    main()
