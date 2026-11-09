"""Central configuration for the AI-Powered Traffic Flow Optimization System."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SNAPSHOT_DIR = BASE_DIR / "static" / "outputs"
DATABASE = BASE_DIR / "database" / "traffic.db"
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# ---- Detection ---------------------------------------------------------
_local_model = BASE_DIR / "models" / "yolov8n.pt"
MODEL_PATH = os.environ.get("YOLO_MODEL", str(_local_model) if _local_model.exists() else "yolov8n.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.35))
FRAME_WIDTH = 640            # frames are resized to this width before detection
PROCESS_FPS = float(os.environ.get("PROCESS_FPS", 5))   # analysed frames per video-second
USE_TRACKING = True          # ByteTrack IDs -> unique vehicle counts (falls back automatically)

# COCO class ids -> vehicle labels
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "bicycle"]
# Passenger Car Units: heavier vehicles load a junction more than a motorcycle does
PCU = {"car": 1.0, "motorcycle": 0.5, "bus": 2.5, "truck": 2.5, "bicycle": 0.3}

# ---- Density thresholds (PCU visible in one frame) ----------------------
LOW_MAX_PCU = 8
MEDIUM_MAX_PCU = 18

# ---- Prediction ----------------------------------------------------------
BUCKET_SECONDS = 2.0         # time resolution of the traffic time-series
HORIZON_STEPS = 5            # predict 5 buckets (~10 s of video time) ahead
WINDOW_STEPS = 6             # history window fed to the ML model
MIN_ML_WINDOWS = 30          # training windows needed before the ML model is used

# ---- Signal optimisation -------------------------------------------------
ROADS = ["North", "East", "South", "West"]   # signal phase order
CYCLE_TIME = 120             # seconds, whole cycle incl. yellow
YELLOW_TIME = 3
MIN_GREEN = 10               # every road ALWAYS gets at least this (no starvation)
MAX_GREEN = 60

# ---- Upload ---------------------------------------------------------------
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
MAX_UPLOAD_MB = 500
