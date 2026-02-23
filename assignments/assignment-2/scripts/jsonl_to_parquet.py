from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e
    return rows


def normalize_detection(det: dict[str, Any]) -> dict[str, Any]:
    """
    Produce per-detection dicts that match the assignment-required fields:
      - class_label
      - confidence_score
      - bounding_box [x_min, y_min, x_max, y_max]
    """
    bbox = det.get("bbox_xyxy") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        bbox = [None, None, None, None]

    out_bbox = [None, None, None, None]
    if bbox[0] is not None:
        out_bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    return {
        "class_label": str(det.get("class_name", det.get("class_id", "unknown"))),
        "confidence_score": float(det.get("conf", 0.0)),
        "bounding_box": out_bbox,
    }


def jsonl_to_framelevel(
    jsonl_path: Path,
    video_id: str,
    fps: float,
    conf_min: float,
) -> pd.DataFrame:
    """
    One row per frame.
    Keep detections ONLY as JSON (string) in `detections_json`.
    """
    entries = load_jsonl(jsonl_path)

    out_rows: list[dict[str, Any]] = []
    for e in entries:
        frame_index = e.get("frame_index")
        frame_file = e.get("frame_file")
        if frame_index is None or frame_file is None:
            continue

        try:
            fi = int(frame_index)
        except Exception:
            continue

        dets = e.get("detections", []) or []
        normalized: list[dict[str, Any]] = []
        for d in dets:
            nd = normalize_detection(d)
            if nd["confidence_score"] >= conf_min:
                normalized.append(nd)

        ts = fi / fps if fps and fps > 0 else float(fi)

        out_rows.append(
            {
                "video_id": video_id,
                "frame_index": fi,
                "timestamp_sec": float(ts),
                "frame_file": str(frame_file),
                # Store as JSON string for maximum compatibility (CSV/parquet viewers/etc.)
                "detections_json": json.dumps(normalized),
            }
        )

    df = pd.DataFrame(out_rows)

    # Enforce column order (clean + predictable)
    cols = ["video_id", "frame_index", "timestamp_sec", "frame_file", "detections_json"]
    df = df[cols]

    return df


def main():
    JSONL_IN = REPO_ROOT / "outputs" / "detections_50ep_carfiltered.jsonl"
    PARQUET_OUT = REPO_ROOT / "outputs" / "detections_50ep_framelevel.parquet"

    df = jsonl_to_framelevel(
        jsonl_path=JSONL_IN,
        video_id="YcvECxtXoxQ",
        fps=1.0,
        conf_min=0.30,
    )

    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_OUT, index=False)

    print(f"[OK] Read:  {JSONL_IN}")
    print(f"[OK] Wrote: {PARQUET_OUT}")
    print(f"[OK] Rows:  {len(df)}")
    if len(df) > 0:
        print(f"[OK] Example detections_json (row 0): {df.iloc[0]['detections_json'][:200]}...")


if __name__ == "__main__":
    main()