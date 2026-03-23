from pathlib import Path
import cv2
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/best.pt"
FRAMES_ROOT = Path("frames")
DETECTIONS_ROOT = Path("detections")
DETECTIONS_ROOT.mkdir(exist_ok=True)

# Set to a specific folder name like "drone_video_2" to process only one video.
# Set to None to process all videos.
ONLY_VIDEO = None

DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 1280

# Harder video settings
VIDEO2_CONF = 0.4
VIDEO2_IMGSZ = 1600
VIDEO2_MAX_BOX_AREA = 12000

model = YOLO(MODEL_PATH)

# -----------------------------
# Helpers
# -----------------------------
def box_area(box):
    x1, y1, x2, y2 = box
    return float(x2 - x1) * float(y2 - y1)

def draw_box(img, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    y_text = max(0, y1 - th - baseline - 4)

    cv2.rectangle(
        img,
        (x1, y_text),
        (x1 + tw + 6, y_text + th + baseline + 4),
        color,
        -1
    )
    cv2.putText(
        img,
        label,
        (x1 + 3, y_text + th + 1),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA
    )

# -----------------------------
# Main loop
# -----------------------------
for video_dir in sorted(FRAMES_ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    if ONLY_VIDEO is not None and video_dir.name != ONLY_VIDEO:
        continue

    out_dir = DETECTIONS_ROOT / video_dir.name
    out_dir.mkdir(exist_ok=True)

    # Default settings
    conf = DEFAULT_CONF
    imgsz = DEFAULT_IMGSZ
    max_box_area = None

    # Override for harder video
    if video_dir.name == "drone_video_2":
        conf = VIDEO2_CONF
        imgsz = VIDEO2_IMGSZ
        max_box_area = VIDEO2_MAX_BOX_AREA

    print(
        f"\nProcessing {video_dir.name} | "
        f"conf={conf} | imgsz={imgsz} | max_box_area={max_box_area}"
    )

    for img_path in sorted(video_dir.glob("*.jpg")):
        print("Processing:", img_path.name)

        results = model(
            str(img_path),
            imgsz=imgsz,
            conf=conf,
            save=False,
            verbose=False
        )

        saved = False

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # -----------------------------
            # Video 2: apply box-area filter
            # -----------------------------
            if max_box_area is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                kept_indices = []
                for i, (box, cls_id) in enumerate(zip(boxes, classes)):
                    class_name = model.names[int(cls_id)]
                    area = box_area(box)

                    if class_name != "drone":
                        continue
                    if area > max_box_area:
                        continue

                    kept_indices.append(i)

                if len(kept_indices) == 0:
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                for i in kept_indices:
                    box = boxes[i]
                    conf_i = confs[i]
                    cls_name = model.names[int(classes[i])]
                    label = f"{cls_name} {conf_i:.2f}"
                    draw_box(image, box, label)

                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), image)
                saved = True
                break

            # -----------------------------
            # Other videos: standard output
            # -----------------------------
            else:
                plotted = r.plot()
                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), plotted)
                saved = True
                break

        if not saved:
            continue

print("Done.")