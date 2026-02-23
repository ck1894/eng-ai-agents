# scripts/debug_retrieval_matching.py

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def safe_len(x: Any) -> int:
    """
    Safely compute "length" for list/array-like objects.
    Returns 0 for None / non-sized objects.
    """
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def get_frame_label_set(dets: Any, do_normalize: bool) -> set[str]:
    """
    dets should be list-like of dict/struct-like items.
    We'll iterate if possible, and try common label keys.
    """
    out: set[str] = set()
    if dets is None:
        return out

    # If dets is not iterable, bail out
    try:
        iterator = iter(dets)
    except Exception:
        return out

    for d in iterator:
        lbl = None

        if isinstance(d, dict):
            lbl = d.get("class_label") or d.get("label") or d.get("name")
        else:
            # Some parquet readers yield row/struct-like objects
            try:
                lbl = d.get("class_label") or d.get("label") or d.get("name")
            except Exception:
                lbl = None

        if lbl is None:
            continue

        s = str(lbl)
        out.add(normalize_label(s) if do_normalize else s)

    return out


def compute_need(n_query_labels: int, min_fraction: float, min_need: int, cap_need: int | None) -> int:
    if n_query_labels <= 0:
        return 0
    need = math.ceil(min_fraction * n_query_labels)
    need = max(min_need, need)
    if cap_need is not None:
        need = min(need, cap_need)
    return need


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-parquet", type=str, default="outputs/detections_50ep_framelevel.parquet")
    parser.add_argument("--model", type=str, default="models/best_50ep.pt")
    parser.add_argument("--hf-dataset", type=str, default="aegean-ai/rav4-exterior-images")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--query-index", type=int, default=0)

    # Matching rule knobs
    parser.add_argument("--min-fraction", type=float, default=0.4)  # 40%
    parser.add_argument("--min-need", type=int, default=1)          # at least 1 label
    parser.add_argument("--cap-need", type=int, default=None)       # e.g. 2 to cap strictness

    # Query label selection knobs
    parser.add_argument("--conf", type=float, default=0.30)         # YOLO conf threshold for query detections
    parser.add_argument("--topk", type=int, default=None)           # keep only top-k labels by confidence

    # Normalization knob
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    # Optional: limit scanning frames for speed
    parser.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    frame_parquet = (REPO_ROOT / args.frame_parquet).resolve()
    model_path = (REPO_ROOT / args.model).resolve()

    if not frame_parquet.exists():
        raise FileNotFoundError(f"Missing frame parquet: {frame_parquet}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    print(f"[DBG] frame_parquet: {frame_parquet}")
    print(f"[DBG] model_path: {model_path}")
    print(f"[DBG] normalize: {args.normalize}")
    print(f"[DBG] rule: min_fraction={args.min_fraction}, min_need={args.min_need}, cap_need={args.cap_need}")
    print(f"[DBG] query: dataset={args.hf_dataset} split={args.split} index={args.query_index}")
    print(f"[DBG] query YOLO: conf={args.conf}, topk={args.topk}")

    # ---- Load frames parquet ----
    df_frames = pd.read_parquet(frame_parquet)
    if "detections" not in df_frames.columns:
        raise ValueError(f"'detections' not found. Columns: {list(df_frames.columns)}")

    total_frames = len(df_frames)
    det_lens = df_frames["detections"].apply(safe_len)
    frames_with_dets = int((det_lens > 0).sum())

    print(f"\n[DBG] total_frames={total_frames}, frames_with_detections={frames_with_dets}")
    print(f"[DBG] detections length stats: min={det_lens.min()} max={det_lens.max()} mean={det_lens.mean():.3f}")

    # Optional: subset frames for speed
    if args.max_frames is not None:
        df_frames = df_frames.iloc[: args.max_frames].copy()
        print(f"[DBG] limiting to max_frames={args.max_frames} -> using {len(df_frames)} frames")

    # Precompute per-frame label sets
    frame_label_sets: list[set[str]] = []
    frame_label_sizes: list[int] = []
    union_frame_labels: set[str] = set()

    for dets in df_frames["detections"].tolist():
        s = get_frame_label_set(dets, do_normalize=args.normalize)
        frame_label_sets.append(s)
        frame_label_sizes.append(len(s))
        union_frame_labels |= s

    print(f"[DBG] unique labels in frames (union): {len(union_frame_labels)}")
    print(f"[DBG] sample frame labels: {sorted(list(union_frame_labels))[:30]}")

    size_counts = Counter(frame_label_sizes)
    print(f"[DBG] frame label-set size distribution (size -> count): {dict(size_counts.most_common(10))}")

    # ---- Load query dataset and run YOLO on a single query image ----
    ds = load_dataset(args.hf_dataset, split=args.split)
    if not (0 <= args.query_index < len(ds)):
        raise IndexError(f"query-index out of range: {args.query_index} (dataset size {len(ds)})")

    img = ds[args.query_index]["image"]  # PIL Image
    model = YOLO(str(model_path))

    results = model.predict(img, conf=args.conf, verbose=False)
    if not results:
        print("\n[DBG] No YOLO results for this query image.")
        return

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or boxes.cls is None:
        print("\n[DBG] No boxes/cls for this query image.")
        return

    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(cls_ids)

    label_to_best_conf: dict[str, float] = {}
    for cid_raw, conf in zip(cls_ids, confs):
        try:
            cid = int(cid_raw)
        except Exception:
            continue

        name = r0.names.get(cid, str(cid))
        label = normalize_label(name) if args.normalize else str(name)

        c = float(conf) if conf is not None else 0.0
        prev = label_to_best_conf.get(label)
        if prev is None or c > prev:
            label_to_best_conf[label] = c

    if not label_to_best_conf:
        print("\n[DBG] Query has 0 labels after conversion.")
        return

    sorted_labels = sorted(label_to_best_conf.items(), key=lambda kv: kv[1], reverse=True)
    if args.topk is not None:
        sorted_labels = sorted_labels[: args.topk]

    query_labels = [lbl for (lbl, _) in sorted_labels]
    query_set = set(query_labels)

    print("\nQUERY LABELS (this query only):")
    for lbl, c in sorted_labels:
        print(f"  - {lbl:20s} conf={c:.3f}")
    print(f"[DBG] query_label_count={len(query_set)}")

    # Sanity: union overlap
    union_overlap = query_set & union_frame_labels
    print(f"\n[DBG] overlap(query_labels, union_frame_labels) size={len(union_overlap)}")
    print(f"[DBG] overlap labels: {sorted(list(union_overlap))}")

    # ---- Compute best achievable overlap with ANY single frame ----
    best_overlap = -1
    best_frame_idx = None
    best_frame_overlap_labels: list[str] = []

    overlaps_per_frame: list[int] = []
    for i, fset in enumerate(frame_label_sets):
        ov = len(query_set & fset)
        overlaps_per_frame.append(ov)
        if ov > best_overlap:
            best_overlap = ov
            best_frame_idx = i
            best_frame_overlap_labels = sorted(list(query_set & fset))

    print(f"\n[DBG] BEST overlap with any frame = {best_overlap} (frame_row_index={best_frame_idx})")
    print(f"[DBG] labels overlapping at best frame: {best_frame_overlap_labels}")

    # ---- Apply rule and count matches ----
    need = compute_need(len(query_set), args.min_fraction, args.min_need, args.cap_need)
    match_frame_indices = [i for i, ov in enumerate(overlaps_per_frame) if ov >= need]

    print(f"\n[DBG] need={need} (from min_fraction/min_need/cap_need)")
    print(f"[DBG] matched_frames={len(match_frame_indices)} out of {len(frame_label_sets)}")

    if match_frame_indices:
        print("\n[DBG] first 10 matched frames (row_index, overlap_count, overlap_labels):")
        for i in match_frame_indices[:10]:
            ov_labels = sorted(list(query_set & frame_label_sets[i]))
            print(f"  - frame_row={i:4d} overlap={len(ov_labels)} labels={ov_labels}")

        time_col = None
        for cand in ["timestamp_sec", "timestamp", "time_sec"]:
            if cand in df_frames.columns:
                time_col = cand
                break

        if time_col is not None:
            print(f"\n[DBG] matched frame times via column '{time_col}':")
            for i in match_frame_indices[:10]:
                print(f"  - frame_row={i:4d} {time_col}={df_frames.iloc[i][time_col]}")
    else:
        print("\n[DBG] No frames matched this rule for this query.")
        print("      If BEST overlap is 0 -> frame label extraction/filtering issue.")
        print("      If BEST overlap > 0 but need is higher -> rule too strict OR query has too many labels.")
        print("      Try: --min-fraction 0.0 --min-need 1  (need=1) or --topk 2")

    ov_counts = Counter(overlaps_per_frame)
    print(f"\n[DBG] overlap distribution (overlap -> count): {dict(ov_counts.most_common(10))}")


if __name__ == "__main__":
    main()