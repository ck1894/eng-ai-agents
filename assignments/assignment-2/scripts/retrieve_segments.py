# scripts/retrieve_segments.py
#
# End-to-end retrieval script:
# - Reads frame-level detections parquet (one row per frame)
#   * supports either:
#       - detections (list of dicts)
#       - detections_json (JSON string or list/dict)
# - Runs YOLO on each query image to get query labels
# - Matches frames where overlap(query_labels, frame_labels) >= need
# - Groups matched frames into contiguous time segments (with optional small gaps)
# - Writes segments parquet
#
# Output columns (segment-level):
# - query_index
# - query_labels               (ALL labels detected in query, after topk; pipe-separated)
# - start_timestamp
# - end_timestamp
# - class_label                (subset of query_labels that actually matched in this segment; pipe-separated)
# - number_of_supporting_detections (sum over frames of |overlap(query_labels, frame_labels)|)
# - youtube_url
#
# Run (example):
#   python scripts/retrieve_segments.py --topk 3 --min-need 2 --min-fraction 0.6 --min-segment-sec 3
#
# Output:
#   outputs/retrieval_segments.parquet

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Helpers
# -----------------------------
def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def safe_len(x: Any) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def parse_detections_maybe_json(x: Any) -> list[dict[str, Any]]:
    """
    Supports:
    - already a list of dicts
    - a dict
    - a JSON string representing list/dict
    - None / empty string -> []
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def get_frame_label_set(dets: Any, do_normalize: bool) -> set[str]:
    """
    dets is list-like of dict/struct-like objects.
    We try common keys for label names.
    """
    out: set[str] = set()
    if dets is None:
        return out

    try:
        iterator = iter(dets)
    except Exception:
        return out

    for d in iterator:
        lbl = None
        if isinstance(d, dict):
            lbl = d.get("class_label") or d.get("label") or d.get("name")
        else:
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


def group_times_into_segments(times: list[float], max_gap_sec: float) -> list[tuple[float, float]]:
    """
    Given times (seconds), group into segments allowing gaps up to max_gap_sec.
    Assumes 1 FPS-ish, but works generally.
    """
    if not times:
        return []

    times = sorted(times)
    segs: list[tuple[float, float]] = []

    start = times[0]
    prev = times[0]

    for t in times[1:]:
        if (t - prev) <= max_gap_sec:
            prev = t
        else:
            segs.append((start, prev))
            start = t
            prev = t

    segs.append((start, prev))
    return segs


def youtube_embed_url(video_id: str, start_sec: int, end_sec: int) -> str:
    if end_sec < start_sec:
        end_sec = start_sec
    return f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}"


def segment_labels_and_support(
    segment_frame_rows: list[int],
    query_set: set[str],
    frame_label_sets: list[set[str]],
) -> tuple[str, int]:
    """
    Return:
      - class_label: ALL labels used for matching within this segment (pipe-separated)
      - support: total number of matched labels across all frames (sum of overlap sizes)

    labels_used = union over frames of (query_set ∩ frame_labels)
    support = Σ over frames |query_set ∩ frame_labels|
    """
    labels_used: set[str] = set()
    support = 0

    for r in segment_frame_rows:
        ov = query_set & frame_label_sets[r]
        if ov:
            labels_used |= ov
            support += len(ov)

    if not labels_used:
        fallback = sorted(list(query_set))[0] if query_set else "unknown"
        return fallback, 0

    label_str = "|".join(sorted(labels_used))
    return label_str, int(support)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--frame-parquet", type=str, default="outputs/detections_50ep_framelevel.parquet")
    parser.add_argument("--model", type=str, default="models/best_50ep.pt")
    parser.add_argument("--hf-dataset", type=str, default="aegean-ai/rav4-exterior-images")
    parser.add_argument("--split", type=str, default="train")

    # Video metadata
    parser.add_argument("--video-id", type=str, default="YcvECxtXoxQ")

    # Query detection knobs
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)

    # Matching rule knobs
    parser.add_argument("--min-fraction", type=float, default=0.40)
    parser.add_argument("--min-need", type=int, default=1)
    parser.add_argument("--cap-need", type=int, default=None)

    # Segment grouping knobs
    parser.add_argument("--max-gap-sec", type=float, default=1.0)
    parser.add_argument("--min-segment-sec", type=float, default=2.0)  # inclusive duration at 1 FPS
    parser.add_argument("--pad-end-sec", type=int, default=0)

    # Normalization
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    # Output
    parser.add_argument("--out-parquet", type=str, default="outputs/retrieval_segments.parquet")

    # Debug
    parser.add_argument("--debug-every", type=int, default=1)

    args = parser.parse_args()

    frame_parquet = (REPO_ROOT / args.frame_parquet).resolve()
    model_path = (REPO_ROOT / args.model).resolve()
    out_path = (REPO_ROOT / args.out_parquet).resolve()

    if not frame_parquet.exists():
        raise FileNotFoundError(f"Missing frame parquet: {frame_parquet}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    print(f"[DBG] frame_parquet: {frame_parquet}")
    print(f"[DBG] model_path: {model_path}")
    print(f"[DBG] out_parquet: {out_path}")
    print(f"[DBG] normalize: {args.normalize}")
    print(f"[DBG] rule: min_fraction={args.min_fraction}, min_need={args.min_need}, cap_need={args.cap_need}")
    print(
        f"[DBG] grouping: max_gap_sec={args.max_gap_sec}, "
        f"min_segment_sec={args.min_segment_sec}, pad_end_sec={args.pad_end_sec}"
    )
    print(f"[DBG] query YOLO: conf={args.conf}, topk={args.topk}")

    # ---- Load frames parquet ----
    df_frames = pd.read_parquet(frame_parquet)

    # Accept either "detections" or "detections_json"
    if "detections" in df_frames.columns:
        det_col = "detections"
    elif "detections_json" in df_frames.columns:
        df_frames["detections"] = df_frames["detections_json"].apply(parse_detections_maybe_json)
        det_col = "detections"
    else:
        raise ValueError(
            f"Expected 'detections' or 'detections_json'. Columns: {list(df_frames.columns)}"
        )

    # Identify time column
    time_col = None
    for cand in ["timestamp_sec", "timestamp", "time_sec"]:
        if cand in df_frames.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = "__row_index_time__"
        df_frames[time_col] = list(range(len(df_frames)))

    det_lens = df_frames[det_col].apply(safe_len)
    frames_with_dets = int((det_lens > 0).sum())

    print(f"[DBG] total_frames={len(df_frames)} frames_with_detections={frames_with_dets}")
    print(f"[DBG] detections length stats: min={det_lens.min()} max={det_lens.max()} mean={det_lens.mean():.3f}")
    print(f"[DBG] using time column: {time_col}")

    # Precompute label sets for each frame
    frame_label_sets: list[set[str]] = []
    for dets in df_frames[det_col].tolist():
        frame_label_sets.append(get_frame_label_set(dets, do_normalize=args.normalize))

    # ---- Load query dataset ----
    ds = load_dataset(args.hf_dataset, split=args.split)
    n_queries = len(ds)
    if args.max_queries is not None:
        n_queries = min(n_queries, args.max_queries)

    print(f"[DBG] queries total available: {len(ds)}")
    print(f"[DBG] queries to process: {n_queries}")

    # ---- YOLO model ----
    model = YOLO(str(model_path))

    # ---- Build segments ----
    out_rows: list[dict[str, Any]] = []
    queries_with_det = 0
    queries_with_match = 0
    total_segments = 0

    for qi in range(n_queries):
        img = ds[qi]["image"]

        results = model.predict(img, conf=args.conf, verbose=False)
        if not results:
            if (qi % args.debug_every) == 0:
                print(f"[{qi:03d}] labels=0")
            continue

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.cls is None:
            if (qi % args.debug_every) == 0:
                print(f"[{qi:03d}] labels=0")
            continue

        cls_ids = boxes.cls.tolist()
        confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(cls_ids)

        # label -> best_conf (max confidence per label)
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
            if (qi % args.debug_every) == 0:
                print(f"[{qi:03d}] labels=0")
            continue

        queries_with_det += 1

        # Top-k label selection
        sorted_labels = sorted(label_to_best_conf.items(), key=lambda kv: kv[1], reverse=True)
        if args.topk is not None:
            sorted_labels = sorted_labels[: args.topk]

        query_labels = sorted([lbl for (lbl, _) in sorted_labels])  # stable order for output
        query_set = set(query_labels)
        need = compute_need(len(query_set), args.min_fraction, args.min_need, args.cap_need)

        matched_rows: list[int] = []
        matched_times: list[float] = []

        for row_i, fset in enumerate(frame_label_sets):
            if len(query_set & fset) >= need:
                matched_rows.append(row_i)
                matched_times.append(float(df_frames.iloc[row_i][time_col]))

        if (qi % args.debug_every) == 0:
            print(f"[{qi:03d}] labels={sorted(list(query_set))} need={need} matches={len(matched_rows)}")

        if not matched_rows:
            continue

        queries_with_match += 1

        segments = group_times_into_segments(matched_times, max_gap_sec=args.max_gap_sec)

        time_to_rows: dict[float, list[int]] = defaultdict(list)
        for r, t in zip(matched_rows, matched_times):
            time_to_rows[t].append(r)

        for (start_t, end_t) in segments:
            # inclusive duration for 1 FPS (e.g., 10..12 is 3 seconds)
            duration = float(end_t - start_t) + 1.0
            if duration < args.min_segment_sec:
                continue

            seg_rows: list[int] = []
            for t in sorted(time_to_rows.keys()):
                if start_t <= t <= end_t:
                    seg_rows.extend(time_to_rows[t])

            if not seg_rows:
                continue

            seg_label, support = segment_labels_and_support(seg_rows, query_set, frame_label_sets)

            start_sec = int(math.floor(start_t))
            end_sec = int(math.ceil(end_t)) + int(args.pad_end_sec)

            out_rows.append(
                {
                    "query_index": qi,
                    "query_labels": "|".join(query_labels),
                    "start_timestamp": float(start_t),
                    "end_timestamp": float(end_t),
                    "class_label": seg_label,
                    "number_of_supporting_detections": int(support),
                    "youtube_url": youtube_embed_url(args.video_id, start_sec, end_sec),
                }
            )
            total_segments += 1

    out_df = pd.DataFrame(out_rows)

    print("\n[DBG] queries total processed:", n_queries)
    print("[DBG] queries with >=1 detection:", queries_with_det)
    print("[DBG] queries with >=1 matching frame:", queries_with_match)
    print("[DBG] total segments:", total_segments)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(f"[OK] output rows (segments): {len(out_df)}")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()