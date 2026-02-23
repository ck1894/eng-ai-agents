import json
from pathlib import Path

p = Path("outputs/detections_50ep_carfiltered.jsonl")
n = 0
with_car = 0
with p.open("r", encoding="utf-8") as f:
    for line in f:
        n += 1
        rec = json.loads(line)
        if "car_bbox_xyxy" in rec:
            with_car += 1

print("total frames:", n)
print("frames with car_bbox_xyxy:", with_car)