from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_label(s: str) -> str:
    return str(s).strip().lower()


def main() -> None:
    # --- paths (adjust if your filenames differ) ---
    FRAME_PARQUET = REPO_ROOT / "outputs" / "detections_50ep_framelevel.parquet"
    MODEL_PATH = REPO_ROOT / "models" / "best_50ep.pt"

    if not FRAME_PARQUET.exists():
        raise FileNotFoundError(f"Missing frame parquet: {FRAME_PARQUET}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")

    # --- load frame-level detections parquet ---
    df_frames = pd.read_parquet(FRAME_PARQUET)

    if "detections" not in df_frames.columns:
        raise ValueError(f"'detections' column not found in {FRAME_PARQUET}. Columns: {list(df_frames.columns)}")

    frame_labels_raw: set[str] = set()
    frame_labels_norm: set[str] = set()

    none_rows = 0
    for row in df_frames["detections"].tolist():
        if row is None:
            none_rows += 1
            continue

        # row should be a list[dict]
        for d in row:
            if not isinstance(d, dict):
                continue
            lbl = d.get("class_label")
            if lbl is None:
                continue
            frame_labels_raw.add(str(lbl))
            frame_labels_norm.add(normalize_label(lbl))

    print(f"[DBG] frames rows: {len(df_frames)}")
    print(f"[DBG] frames rows with detections=None: {none_rows}")
    print("\nFRAME LABELS (raw):")
    print(sorted(frame_labels_raw))
    print("\nFRAME LABELS (normalized):")
    print(sorted(frame_labels_norm))

    # --- run model on query dataset images ---
    model = YOLO(str(MODEL_PATH))

    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")

    query_labels_raw: set[str] = set()
    query_labels_norm: set[str] = set()
    queries_with_det = 0

    for i in range(len(ds)):
        img = ds[i]["image"]  # PIL Image
        results = model.predict(img, verbose=False)

        if not results:
            continue

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.cls is None:
            continue

        cls_ids = boxes.cls.tolist()
        confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(cls_ids)

        # grab labels (no thresholding here; we just want to see label names)
        labels_this = set()
        for cls_id, conf in zip(cls_ids, confs):
            try:
                cid = int(cls_id)
            except Exception:
                continue
            name = r0.names.get(cid, str(cid))
            labels_this.add(str(name))

        if labels_this:
            queries_with_det += 1
            for l in labels_this:
                query_labels_raw.add(l)
                query_labels_norm.add(normalize_label(l))

    print(f"\n[DBG] queries total: {len(ds)}")
    print(f"[DBG] queries with >=1 detection: {queries_with_det}")

    print("\nQUERY LABELS (raw):")
    print(sorted(query_labels_raw))
    print("\nQUERY LABELS (normalized):")
    print(sorted(query_labels_norm))

    # --- overlap check ---
    overlap_raw = frame_labels_raw.intersection(query_labels_raw)
    overlap_norm = frame_labels_norm.intersection(query_labels_norm)

    print("\nOVERLAP (raw):")
    print(sorted(overlap_raw))
    print("\nOVERLAP (normalized):")
    print(sorted(overlap_norm))


if __name__ == "__main__":
    main()