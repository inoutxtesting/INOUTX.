# ============================================================
# INOUTX - VEHICLE DETECTION + TRACKING
# ============================================================

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "yolov8n.pt")

# COCO class 5 = bus
# Strict vehicle filtering: Only detect and track buses (COCO class ID 5).
# Cars (2), motorcycles (3), trucks (7), and all other classes are strictly ignored.
BUS_CLASS_ID = 5
VEHICLE_CLASSES = [BUS_CLASS_ID]

# Lower = detects more distant vehicles
DETECTION_CONFIDENCE = 0.20
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640


# ============================================================
# INDEPENDENT VEHICLE TRACKER CLASS
# ============================================================

class VehicleTracker:
    """
    Encapsulates an independent YOLO ByteTrack instance per camera.
    Configured to strictly track only buses (COCO class ID 5).
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=DETECTION_CONFIDENCE,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        classes=None,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        # Enforce bus class ID 5
        self.classes = [BUS_CLASS_ID]
        self.model = YOLO(self.model_path)

    def track(self, frame):
        """
        Detect and track vehicles in the provided frame.
        ByteTrack maintains persistent IDs across frames for this camera.
        """
        if frame is None:
            return []

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )
            return results
        except Exception as e:
            print("YOLO TRACKING ERROR:", e)
            return []


# ============================================================
# GLOBAL DEFAULT TRACKER (For backward compatibility)
# ============================================================

_global_tracker = None


def get_default_tracker() -> VehicleTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = VehicleTracker()
    return _global_tracker


def track_bus(frame):
    """Legacy backward-compatible tracking function."""
    return get_default_tracker().track(frame)
