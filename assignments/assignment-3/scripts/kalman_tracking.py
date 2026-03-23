from pathlib import Path
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/best.pt"

FRAMES_ROOT = Path("frames")
VIDEOS_ROOT = Path("videos")          # expects drone_video_1.mp4, drone_video_2.mp4, etc.
TRACKING_FRAMES_ROOT = Path("tracking_frames")
OUTPUT_VIDEOS_ROOT = Path("tracking_videos")

TRACKING_FRAMES_ROOT.mkdir(exist_ok=True)
OUTPUT_VIDEOS_ROOT.mkdir(exist_ok=True)

ONLY_VIDEO = None   # set to "drone_video_2" to run only one video

DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 1280

VIDEO2_CONF = 0.4
VIDEO2_IMGSZ = 1600
VIDEO2_MAX_BOX_AREA = 12000

DEFAULT_FPS = 30
MAX_MISSING = 3  # keep saving frames for short missing gaps while track is alive

# Kalman tuning
PROCESS_NOISE = 5.0
MEASUREMENT_NOISE = 20.0
INITIAL_P = 500.0

model = YOLO(MODEL_PATH)

# -----------------------------
# Helpers
# -----------------------------
def box_area(box):
    x1, y1, x2, y2 = box
    return float(x2 - x1) * float(y2 - y1)

def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

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

def draw_trajectory(img, trajectory, color=(0, 0, 255), thickness=2):
    if len(trajectory) >= 2:
        pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

    if len(trajectory) >= 1:
        x, y = map(int, trajectory[-1])
        cv2.circle(img, (x, y), 4, color, -1)

def make_kalman_filter(x, y):
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State: [x, y, vx, vy]
    kf.x = np.array([x, y, 0.0, 0.0], dtype=np.float32)

    # State transition
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # Measurement function: observe [x, y]
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Covariances
    kf.P *= INITIAL_P
    kf.R = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kf.Q = np.eye(4, dtype=np.float32) * PROCESS_NOISE

    return kf

def get_video_fps(video_name):
    video_path = VIDEOS_ROOT / f"{video_name}.mp4"
    if not video_path.exists():
        return DEFAULT_FPS

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 1:
        return DEFAULT_FPS
    return fps

def build_output_video(frames_dir, output_path, fps):
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print(f"No frames found for {frames_dir.name}, skipping video creation.")
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def get_best_detection(results, video_name):
    """
    Reuses Task 1 logic, but returns a single accepted box:
    - for normal videos: highest-confidence drone box
    - for drone_video_2: highest-confidence drone box that also passes max area filter
    """
    if not results:
        return None

    conf = DEFAULT_CONF
    imgsz = DEFAULT_IMGSZ
    max_box_area = None

    if video_name == "drone_video_2":
        conf = VIDEO2_CONF
        imgsz = VIDEO2_IMGSZ
        max_box_area = VIDEO2_MAX_BOX_AREA

    # results already computed with proper conf/imgsz outside
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    candidates = []
    for box, conf_i, cls_id in zip(boxes, confs, classes):
        class_name = model.names[int(cls_id)]
        if class_name != "drone":
            continue

        if max_box_area is not None and box_area(box) > max_box_area:
            continue

        candidates.append((float(conf_i), box))

    if not candidates:
        return None

    # choose highest-confidence accepted box
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_conf, best_box = candidates[0]
    return best_box, best_conf

# -----------------------------
# Main loop
# -----------------------------
for video_dir in sorted(FRAMES_ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    if ONLY_VIDEO is not None and video_dir.name != ONLY_VIDEO:
        continue

    # Per-video detector settings (same as Task 1)
    conf = DEFAULT_CONF
    imgsz = DEFAULT_IMGSZ

    if video_dir.name == "drone_video_2":
        conf = VIDEO2_CONF
        imgsz = VIDEO2_IMGSZ

    print(f"\nProcessing {video_dir.name} | conf={conf} | imgsz={imgsz}")

    out_frames_dir = TRACKING_FRAMES_ROOT / video_dir.name
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    # clean old jpgs from this video's tracking folder
    for old_file in out_frames_dir.glob("*.jpg"):
        old_file.unlink()

    kf = None
    trajectory = []
    missing_count = 0
    saved_idx = 0

    frame_paths = sorted(video_dir.glob("*.jpg"))

    for img_path in frame_paths:
        print("Processing:", img_path.name)

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # 1) Run detector using same logic as Task 1
        results = model(
            str(img_path),
            imgsz=imgsz,
            conf=conf,
            save=False,
            verbose=False
        )

        detection = get_best_detection(results, video_dir.name)

        det_box = None
        det_conf = None
        det_center = None

        if detection is not None:
            det_box, det_conf = detection
            det_center = box_center(det_box)

        # 2) Initialize tracker on first valid detection
        if kf is None:
            if det_center is None:
                continue  # no track yet, and no detection -> skip frame entirely
            kf = make_kalman_filter(det_center[0], det_center[1])
            missing_count = 0

        else:
            # 3) Predict every frame once tracker exists
            kf.predict()

        # 4) Update if detection exists, otherwise keep predicting
        if det_center is not None:
            kf.update(det_center)
            missing_count = 0
        else:
            missing_count += 1

        # 5) Decide whether drone is still considered present
        track_alive = (kf is not None) and (missing_count <= MAX_MISSING)

        if not track_alive:
            # end current track until reacquired
            kf = None
            trajectory = []
            continue

        # 6) Save only frames where drone is present
        est_x, est_y = float(kf.x[0]), float(kf.x[1])
        trajectory.append((int(est_x), int(est_y)))

        # Overlay detector box if present
        if det_box is not None:
            draw_box(image, det_box, f"drone {det_conf:.2f}", color=(0, 255, 0), thickness=2)

        # Overlay tracker trajectory
        draw_trajectory(image, trajectory, color=(0, 0, 255), thickness=2)

        saved_idx += 1
        out_path = out_frames_dir / f"{saved_idx:06d}.jpg"
        cv2.imwrite(str(out_path), image)

    # 7) Build one output video per input video
    fps = get_video_fps(video_dir.name)
    output_video_path = OUTPUT_VIDEOS_ROOT / f"{video_dir.name}_tracking.mp4"
    build_output_video(out_frames_dir, output_video_path, fps)
    print(f"Saved video: {output_video_path}")

print("Done.")