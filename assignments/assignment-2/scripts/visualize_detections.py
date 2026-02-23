from __future__ import annotations

import json
import random
from pathlib import Path
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_detections_jsonl(jsonl_path: Path):
    """
    Load JSONL file (one JSON object per line).
    """
    entries = []
    with jsonl_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e
    return entries


def draw_one_image(
    image_path: Path,
    detections: list[dict],
    out_path: Path,
    conf_min: float = 0.25,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for det in detections:
        conf = float(det.get("conf", 0.0))
        if conf < conf_min:
            continue

        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        label = det.get("class_name", str(det.get("class_id", "unknown")))
        text = f"{label} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    # Use 50 epoch detections
    detections_jsonl = REPO_ROOT / "outputs" / "detections_50ep.jsonl"
    frames_dir = REPO_ROOT / "data" / "frames"

    # Output folder for 50 epoch experiment
    out_dir = REPO_ROOT / "outputs" / "viz_50ep"

    conf_min = 0.25
    num_random = 12
    num_with_dets = 12

    if not detections_jsonl.exists():
        raise FileNotFoundError(f"Detections file not found: {detections_jsonl}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    per_frame = load_detections_jsonl(detections_jsonl)

    all_entries = per_frame

    with_dets = []
    for e in per_frame:
        dets = e.get("detections", []) or []
        keep = [d for d in dets if float(d.get("conf", 0.0)) >= conf_min]
        if keep:
            with_dets.append(e)

    print(f"Total frames in JSONL: {len(all_entries)}")
    print(f"Frames with detections (conf >= {conf_min}): {len(with_dets)}")
    print(f"Saving visualizations to: {out_dir}")

    random_entries = random.sample(all_entries, min(num_random, len(all_entries)))
    det_entries = random.sample(with_dets, min(num_with_dets, len(with_dets))) if with_dets else []

    # Random frames
    for e in random_entries:
        frame_file = e.get("frame_file")
        if not frame_file:
            continue
        img_path = frames_dir / frame_file
        out_path = out_dir / "random" / frame_file
        draw_one_image(img_path, e.get("detections", []) or [], out_path, conf_min)

    # Frames with detections
    for e in det_entries:
        frame_file = e.get("frame_file")
        if not frame_file:
            continue
        img_path = frames_dir / frame_file
        out_path = out_dir / "with_dets" / frame_file
        draw_one_image(img_path, e.get("detections", []) or [], out_path, conf_min)

    print("\nDone.")
    print(f"Check: {out_dir}/random/")
    print(f"Check: {out_dir}/with_dets/")


if __name__ == "__main__":
    main()