from ultralytics import YOLO
from pathlib import Path
import json

# Resolve repo root (assignment-2/)
REPO_ROOT = Path(__file__).resolve().parents[1]

FRAMES_DIR = REPO_ROOT / "data" / "frames"
MODEL_PATH = REPO_ROOT / "models" / "best_50ep.pt"
OUT_PATH = REPO_ROOT / "outputs" / "detections_50ep.jsonl"


def main():
    # Safety checks
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames folder not found: {FRAMES_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Using model:", MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"Found {len(frame_paths)} frames in {FRAMES_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w") as f:
        for i, fp in enumerate(frame_paths):
            results = model.predict(source=str(fp), imgsz=640, verbose=False)
            r = results[0]

            names = r.names

            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]

                    dets.append({
                        "class_id": cls_id,
                        "class_name": names[cls_id],
                        "conf": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    })

            row = {
                "frame_file": fp.name,
                "frame_index": i,
                "detections": dets
            }

            f.write(json.dumps(row) + "\n")

            if i % 50 == 0:
                print(f"processed {i}/{len(frame_paths)}")

    print(f"Saved detections to {OUT_PATH}")


if __name__ == "__main__":
    main()