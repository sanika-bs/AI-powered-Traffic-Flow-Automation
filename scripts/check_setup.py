"""Verify the installation and download the YOLO weights (needs internet the first time)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ok = True
for mod in ["flask", "cv2", "numpy", "matplotlib", "sklearn", "ultralytics"]:
    try:
        importlib.import_module(mod)
        print(f"[ OK ] {mod}")
    except ImportError:
        print(f"[FAIL] {mod}  ->  pip install -r requirements.txt")
        ok = False
if ok:
    import config
    from modules.detector import VehicleDetector
    import numpy as np
    print("Loading YOLO model (downloads yolov8n.pt on first run)...")
    det = VehicleDetector()
    det.detect(np.zeros((360, 640, 3), np.uint8))
    print(f"[ OK ] YOLO ready ({config.MODEL_PATH}); tracking = {det.tracking}")
    print("\nAll good. Start the app with:  python app.py")
else:
    sys.exit(1)
