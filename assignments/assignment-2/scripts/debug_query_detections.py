from __future__ import annotations

from pathlib import Path

import cv2
from datasets import load_dataset
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_path = REPO_ROOT / "models" / "best_50ep.pt"
    out_dir = REPO_ROOT / "outputs" / "query_viz"

    conf_min = 0.30
    max_queries = None  # set e.g. 10 if you only want a few

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    model = YOLO(str(model_path))

    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")
    n = len(ds) if max_queries is None else min(len(ds), int(max_queries))

    out_dir.mkdir(parents=True, exist_ok=True)

    num_with_dets = 0

    print(f"[OK] Loaded dataset: {len(ds)} queries")
    print(f"[OK] Processing: {n} queries")
    print(f"[OK] Saving to: {out_dir}")

    for qi in range(n):
        pil_img = ds[qi]["image"]

        res = model.predict(source=pil_img, conf=conf_min, iou=0.5, verbose=False)[0]

        labels: list[str] = []
        if res.boxes is not None and len(res.boxes) > 0:
            num_with_dets += 1
            names = model.model.names
            for b in res.boxes:
                if float(b.conf.item()) >= conf_min:
                    cls_id = int(b.cls.item())
                    labels.append(str(names.get(cls_id, str(cls_id))))

        seen = set()
        uniq_labels = []
        for x in labels:
            if x not in seen:
                uniq_labels.append(x)
                seen.add(x)

        out_path = out_dir / f"query_{qi:03d}.jpg"
        img_bgr = res.plot()  # numpy array (BGR)
        cv2.imwrite(str(out_path), img_bgr)

        print(f"[Q{qi:02d}] dets={len(labels)} uniq={uniq_labels} saved={out_path.name}")

    print(f"\n[DBG] queries with >=1 detection: {num_with_dets}/{n}")


if __name__ == "__main__":
    main()