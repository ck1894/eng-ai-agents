import json
from pathlib import Path

import cv2
from ultralytics import YOLO

CAR_CONF_MIN = 0.30
PERSON_CONF_MIN = 0.30


def get_class_id(model: YOLO, class_name: str) -> int:
    names = model.model.names  # id -> name
    ids = [i for i, n in names.items() if n == class_name]
    if not ids:
        raise RuntimeError(f"COCO model does not contain class '{class_name}'.")
    return ids[0]


def get_largest_bbox_xyxy(res, target_cls_id: int, conf_min: float):
    best = None  # (area, conf, x1,y1,x2,y2)
    if res.boxes is None:
        return None

    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id != target_cls_id:
            continue

        conf = float(b.conf.item())
        if conf < conf_min:
            continue

        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cand = (area, conf, x1, y1, x2, y2)

        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return None

    _, conf, x1, y1, x2, y2 = best
    return (x1, y1, x2, y2, conf)


def get_all_bboxes_xyxy(res, target_cls_id: int, conf_min: float):
    out = []
    if res.boxes is None:
        return out

    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id != target_cls_id:
            continue

        conf = float(b.conf.item())
        if conf < conf_min:
            continue

        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        out.append((x1, y1, x2, y2, conf))
    return out


def center_xy(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside_box(px, py, box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def shrink_box(box_xyxy, shrink_frac: float):
    # shrink_frac = 0.08 means shrink 8% on each side
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    dx = w * shrink_frac
    dy = h * shrink_frac
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def main():
    frames_dir = Path("data/frames")
    in_jsonl = Path("outputs/detections_50ep.jsonl")
    out_jsonl = Path("outputs/detections_50ep_carfiltered.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    coco = YOLO("yolov8n.pt")
    CAR_ID = get_class_id(coco, "car")
    PERSON_ID = get_class_id(coco, "person")

    # This helps exclude people standing close to the car edge
    CAR_SHRINK_FRAC = 0.08  # try 0.05~0.12

    with in_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            frame_file = rec["frame_file"]
            frame_path = frames_dir / frame_file

            img = cv2.imread(str(frame_path))
            if img is None:
                fout.write(json.dumps(rec) + "\n")
                continue

            coco_res = coco.predict(source=img, conf=min(CAR_CONF_MIN, PERSON_CONF_MIN), verbose=False)[0]

            car = get_largest_bbox_xyxy(coco_res, CAR_ID, CAR_CONF_MIN)
            persons = get_all_bboxes_xyxy(coco_res, PERSON_ID, PERSON_CONF_MIN)

            # If no car detected, keep original (avoid dropping everything)
            if car is None:
                fout.write(json.dumps(rec) + "\n")
                continue

            car_x1, car_y1, car_x2, car_y2, car_conf = car
            car_box = shrink_box((car_x1, car_y1, car_x2, car_y2), CAR_SHRINK_FRAC)

            filtered = []
            for det in rec.get("detections", []) or []:
                bbox = det.get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    continue

                cx, cy = center_xy(bbox)

                # must be inside (shrunken) car box
                if not point_inside_box(cx, cy, car_box):
                    continue

                # must NOT be inside any person box
                inside_person = False
                for px1, py1, px2, py2, _pconf in persons:
                    if point_inside_box(cx, cy, (px1, py1, px2, py2)):
                        inside_person = True
                        break
                if inside_person:
                    continue

                filtered.append(det)

            rec2 = dict(rec)
            rec2["detections"] = filtered
            rec2["car_bbox_xyxy"] = [car_x1, car_y1, car_x2, car_y2]
            rec2["car_conf"] = car_conf
            rec2["car_bbox_xyxy_shrunk"] = [car_box[0], car_box[1], car_box[2], car_box[3]]
            rec2["person_bboxes_xyxy"] = [[p[0], p[1], p[2], p[3]] for p in persons]

            fout.write(json.dumps(rec2) + "\n")

    print(f"[OK] Wrote: {out_jsonl}")


if __name__ == "__main__":
    main()