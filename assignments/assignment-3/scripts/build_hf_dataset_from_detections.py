from pathlib import Path
import pandas as pd

DETECTIONS_ROOT = Path("detections")

FPS = 5
ORIGINAL_FPS = 30
STRIDE = ORIGINAL_FPS // FPS

rows = []

for video_dir in sorted(DETECTIONS_ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    video_name = video_dir.name
    frame_paths = sorted(video_dir.glob("*.jpg"))

    for i, img_path in enumerate(frame_paths):
        if i % STRIDE != 0:
            continue

        rows.append({
            "image": str(img_path),   # path to image
            "video_title": video_name,
            "frame_name": img_path.name,
            "frame_index": i,
            "timestamp_sec": i / ORIGINAL_FPS,
        })

print(f"Collected {len(rows)} detection frames")

# -----------------------------
# Save as Parquet
# -----------------------------
df = pd.DataFrame(rows)
df.to_parquet("detections.parquet", index=False)

print("Saved detections.parquet")