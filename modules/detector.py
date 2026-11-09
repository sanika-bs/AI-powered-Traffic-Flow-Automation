"""YOLOv8 vehicle detection (+ optional ByteTrack IDs for unique vehicle counting)."""
import cv2
import config

COLORS = {  # BGR
    "car": (255, 160, 0), "motorcycle": (0, 200, 255), "bus": (0, 120, 255),
    "truck": (60, 60, 255), "bicycle": (180, 90, 255),
}


class DetectorError(RuntimeError):
    """Raised when the YOLO model cannot be loaded."""


def preprocess(frame, width=None):
    """Resize to a fixed width and apply CLAHE contrast enhancement on luminance."""
    width = width or config.FRAME_WIDTH
    h, w = frame.shape[:2]
    if w != width:
        frame = cv2.resize(frame, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class VehicleDetector:
    """Wraps an Ultralytics YOLOv8 model. One instance per video so tracker state is isolated."""

    def __init__(self, model_path=None, conf=None, use_tracking=None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("The 'ultralytics' package is not installed. Run: pip install ultralytics") from exc
        try:
            self.model = YOLO(model_path or config.MODEL_PATH)
        except Exception as exc:
            raise DetectorError(
                f"Could not load YOLO weights '{model_path or config.MODEL_PATH}'. "
                "The first run downloads yolov8n.pt (needs internet) - or place the file in the models/ folder. "
                f"Details: {exc}") from exc
        self.conf = conf if conf is not None else config.CONF_THRESHOLD
        self.tracking = config.USE_TRACKING if use_tracking is None else use_tracking
        self.class_ids = list(config.VEHICLE_CLASSES)

    def _run(self, frame):
        if self.tracking:
            try:
                return self.model.track(frame, persist=True, conf=self.conf, classes=self.class_ids,
                                        tracker="bytetrack.yaml", verbose=False)[0]
            except Exception:       # tracker dependency missing -> plain detection
                self.tracking = False
        return self.model.predict(frame, conf=self.conf, classes=self.class_ids, verbose=False)[0]

    def detect(self, frame):
        """Return list of dicts: label, conf, box=(x1,y1,x2,y2), id (None without tracking)."""
        result = self._run(frame)
        out = []
        if result.boxes is None:
            return out
        for b in result.boxes:
            label = config.VEHICLE_CLASSES.get(int(b.cls[0]))
            if not label:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
            out.append({"label": label, "conf": float(b.conf[0]), "box": (x1, y1, x2, y2), "id": tid})
        return out


def create_detector():
    return VehicleDetector()


def annotate(frame, detections, title, total, level, pcu):
    """Draw bounding boxes + a heads-up display on a copy of the frame."""
    img = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(d["label"], (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f'{d["label"]} {d["conf"]:.2f}' + (f' #{d["id"]}' if d.get("id") is not None else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, tag, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    lvl_color = {"Low": (80, 200, 80), "Medium": (0, 190, 255), "High": (60, 60, 240)}.get(level, (200, 200, 200))
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 24, 20), -1)
    cv2.putText(img, f"{title} | Vehicles: {total} | PCU: {pcu:.1f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (img.shape[1] - 110, 4), (img.shape[1] - 6, 26), lvl_color, -1)
    cv2.putText(img, level.upper(), (img.shape[1] - 100, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return img
