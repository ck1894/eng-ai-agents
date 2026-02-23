from __future__ import annotations

import json
import random
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_detections_jsonl(jsonl_path: Path):
    entries = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def draw_image_with_detections(
    image_path: Path,
    entry: dict,
    out_path: Path,
    conf_min: float,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Draw car bbox if present
    car_bbox = entry.get("car_bbox_xyxy")
    if isinstance(car_bbox, (list, tuple)) and len(car_bbox) == 4:
        x1, y1, x2, y2 = map(int, car_bbox)
        car_conf = entry.get("car_conf")
        label = "car"
        if car_conf is not None:
            try:
                label = f"car {float(car_conf):.2f}"
            except Exception:
                pass

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    # Draw part detections
    for det in entry.get("detections", []) or []:
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
    detections_jsonl = REPO_ROOT / "outputs" / "detections_50ep_carfiltered.jsonl"
    frames_dir = REPO_ROOT / "data" / "frames"
    out_dir = REPO_ROOT / "outputs" / "viz_50ep_carfiltered"

    conf_min = 0.25
    num_random = 12
    num_with_dets = 12

    if not detections_jsonl.exists():
        raise FileNotFoundError(f"Detections file not found: {detections_jsonl}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    entries = load_detections_jsonl(detections_jsonl)

    with_dets = []
    for e in entries:
        dets = e.get("detections", []) or []
        keep = [d for d in dets if float(d.get("conf", 0.0)) >= conf_min]
        if keep:
            with_dets.append(e)

    print(f"Total frames: {len(entries)}")
    print(f"Frames with detections (conf >= {conf_min}): {len(with_dets)}")
    print(f"Output folder: {out_dir}")

    random_entries = random.sample(entries, min(num_random, len(entries)))
    det_entries = random.sample(with_dets, min(num_with_dets, len(with_dets))) if with_dets else []

    for e in random_entries:
        frame_file = e.get("frame_file")
        if not frame_file:
            continue
        img_path = frames_dir / frame_file
        out_path = out_dir / "random" / frame_file
        draw_image_with_detections(img_path, e, out_path, conf_min)

    for e in det_entries:
        frame_file = e.get("frame_file")
        if not frame_file:
            continue
        img_path = frames_dir / frame_file
        out_path = out_dir / "with_dets" / frame_file
        draw_image_with_detections(img_path, e, out_path, conf_min)

    print("Done.")
    print(f"Check: {out_dir / 'random'}")
    print(f"Check: {out_dir / 'with_dets'}")


if __name__ == "__main__":
    main()