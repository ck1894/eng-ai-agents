from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def to_hashable_str(x: Any) -> str:
    """Convert list/np.ndarray/etc into a stable string so pandas can dedupe/group safely."""
    if x is None:
        return "None"
    # numpy arrays (including object arrays)
    if isinstance(x, np.ndarray):
        try:
            return np.array2string(x, separator=",", threshold=1000)
        except Exception:
            return str(x)
    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        parts = []
        for v in x:
            parts.append(to_hashable_str(v))
        return "[" + ", ".join(parts) + "]"
    # dict
    if isinstance(x, dict):
        items = []
        for k in sorted(x.keys(), key=lambda z: str(z)):
            items.append(f"{to_hashable_str(k)}:{to_hashable_str(x[k])}")
        return "{" + ", ".join(items) + "}"
    return str(x)


def gini_from_counts(counts: np.ndarray) -> float:
    """
    Proper Gini coefficient for nonnegative counts.
    Returns value in [0, 1], where 0 is perfectly even, 1 is maximally concentrated.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return 0.0
    if np.any(counts < 0):
        raise ValueError("Counts must be nonnegative for Gini.")
    s = counts.sum()
    if s == 0:
        return 0.0

    # Sort ascending
    x = np.sort(counts)
    n = x.size
    # Gini formula: (2*sum(i*x_i) / (n*sum(x))) - (n+1)/n
    i = np.arange(1, n + 1)
    g = (2.0 * np.sum(i * x) / (n * s)) - (n + 1) / n
    # Numerical safety
    return float(max(0.0, min(1.0, g)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segments-parquet",
        type=str,
        default=str(REPO_ROOT / "outputs" / "retrieval_segments.parquet"),
    )
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    seg_path = Path(args.segments_parquet)
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segments parquet: {seg_path}")

    df = pd.read_parquet(seg_path)
    print(f"[DBG] loaded: {seg_path}")
    print(f"[DBG] rows (segments): {len(df)}")
    print(f"[DBG] columns: {list(df.columns)}")

    query_col = "query_index"
    if query_col not in df.columns:
        raise ValueError(f"Expected '{query_col}' column in segments parquet.")

    total_segments = len(df)

    per_query = (
        df.groupby(query_col, dropna=False)
        .size()
        .reset_index(name="segment_count")
        .sort_values("segment_count", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== TOP QUERIES BY SEGMENT COUNT ===")
    print(per_query.head(args.topn).to_string(index=False))

    top5 = int(per_query.head(5)["segment_count"].sum())
    top10 = int(per_query.head(10)["segment_count"].sum())
    print("\n=== CONCENTRATION ===")
    print(f"Total segments: {total_segments}")
    print(f"Top 5 queries produce:  {top5} segments ({top5/total_segments:.1%})")
    print(f"Top 10 queries produce: {top10} segments ({top10/total_segments:.1%})")

    g = gini_from_counts(per_query["segment_count"].to_numpy())
    print(f"Gini (0 even, 1 concentrated): {g:.3f}")

    heavy_queries = set(per_query.head(10)[query_col].tolist())
    show = df[df[query_col].isin(heavy_queries)].copy()

    print("\n=== HEAVY QUERY DETAILS (TOP 10) ===")

    if "class_label" in show.columns:
        print("\nTop retrieved labels within TOP 10 queries:")
        print(show["class_label"].value_counts().head(20).to_string())
    else:
        print("\n(No 'class_label' column found; skipping label breakdown.)")

    # Safely print query_labels per heavy query
    if "query_labels" in show.columns:
        tmp = (
            show[[query_col, "query_labels"]]
            .assign(query_labels_str=show["query_labels"].map(to_hashable_str))
            .loc[:, [query_col, "query_labels_str"]]
            .drop_duplicates()
            .sort_values([query_col, "query_labels_str"])
        )
        print("\nQuery label-sets for TOP 10 queries:")
        print(tmp.to_string(index=False))
    else:
        print("\n(No 'query_labels' column found; skipping query-label breakdown.)")

    print("\nDone.")


if __name__ == "__main__":
    main()