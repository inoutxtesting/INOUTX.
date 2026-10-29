# ============================================================
# IN/OUT X - CONFIGURATION
# ============================================================

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_NAME = "IN/OUT X"
VERSION = "2.0 - Dual Camera (KRCT & KRCE)"


# ============================================================
# CREDENTIAL MASKING & URL VALIDATION HELPERS
# ============================================================

def mask_rtsp_url(url: int | str) -> str:
    """
    Mask credentials in RTSP URLs for safe terminal logs and error reports.
    Handles passwords containing '@' signs. If an integer camera index is provided,
    returns a safe camera index description.
    Example: rtsp://user:p@ss@word@192.168.0.1:554/stream -> rtsp://***:***@192.168.0.1:554/stream
    """
    if isinstance(url, int):
        return f"Camera Index {url}"
    if not isinstance(url, str):
        return str(url)
    return re.sub(r"://.*@([0-9a-zA-Z\.\-]+(?::\d+)?/)", r"://***:***@\1", url)


def validate_rtsp_url(url: int | str) -> tuple[bool, str]:
    """
    Retained for backward compatibility. Returns (True, '') without performing checks.
    """
    return True, ""


# ============================================================
# CAMERA 1 CONFIGURATION — KRCT
# ============================================================

CAMERA_1_ID = "KRCT_GATE_01"
COLLEGE_1 = "KRCT"
GATE_NAME_1 = "KRCT Main Gate"
CAMERA_SOURCE_1 = (
    os.getenv("KRCT_RTSP_URL")
    or os.getenv("CAMERA_SOURCE_1")
    or os.getenv("CAMERA_SOURCE")
    or "rtsp://krct:krct@123@192.168.0.141:554/Streaming/channels/101"
)
if isinstance(CAMERA_SOURCE_1, str) and CAMERA_SOURCE_1.strip().isdigit():
    CAMERA_SOURCE_1 = int(CAMERA_SOURCE_1.strip())

from line_calibration import load_calibrated_lines

# Calibrated virtual lines for Camera 1 (KRCT) and Camera 2 (KRCE)
_calibrated = load_calibrated_lines()
EXIT_LINE_1 = _calibrated["EXIT_LINE_1"]
ENTRY_LINE_1 = _calibrated["ENTRY_LINE_1"]


# ============================================================
# CAMERA 2 CONFIGURATION — KRCE
# ============================================================

CAMERA_2_ID = "KRCE_GATE_01"
COLLEGE_2 = "KRCE"
GATE_NAME_2 = "KRCE Main Gate"
CAMERA_SOURCE_2 = (
    os.getenv("KRCE_RTSP_URL")
    or os.getenv("CAMERA_SOURCE_2")
    or "rtsp://krct:Krct@3701@192.168.77.228:554/Streaming/channels/101"
)
if isinstance(CAMERA_SOURCE_2, str) and CAMERA_SOURCE_2.strip().isdigit():
    CAMERA_SOURCE_2 = int(CAMERA_SOURCE_2.strip())

# Calibrated virtual lines for Camera 2 (KRCE)
EXIT_LINE_2 = _calibrated["EXIT_LINE_2"]
ENTRY_LINE_2 = _calibrated["ENTRY_LINE_2"]


def reload_calibrated_lines():
    """Reloads line coordinates from lines_config.json into the registry."""
    global ENTRY_LINE_1, EXIT_LINE_1, ENTRY_LINE_2, EXIT_LINE_2, CAMERAS
    _lines = load_calibrated_lines()
    ENTRY_LINE_1 = _lines["ENTRY_LINE_1"]
    EXIT_LINE_1 = _lines["EXIT_LINE_1"]
    ENTRY_LINE_2 = _lines["ENTRY_LINE_2"]
    EXIT_LINE_2 = _lines["EXIT_LINE_2"]
    if CAMERA_1_ID in CAMERAS:
        CAMERAS[CAMERA_1_ID]["entry_line"] = ENTRY_LINE_1
        CAMERAS[CAMERA_1_ID]["exit_line"] = EXIT_LINE_1
    if CAMERA_2_ID in CAMERAS:
        CAMERAS[CAMERA_2_ID]["entry_line"] = ENTRY_LINE_2
        CAMERAS[CAMERA_2_ID]["exit_line"] = EXIT_LINE_2
    return _lines


# ============================================================
# CENTRALIZED DUAL-CAMERA REGISTRY
# ============================================================

CAMERAS = {
    CAMERA_1_ID: {
        "camera_id": CAMERA_1_ID,
        "college": COLLEGE_1,
        "gate_name": GATE_NAME_1,
        "source": CAMERA_SOURCE_1,
        "entry_line": ENTRY_LINE_1,
        "exit_line": EXIT_LINE_1,
        "enabled": True,
        "ws_path": "/ws/camera1",
    },
    CAMERA_2_ID: {
        "camera_id": CAMERA_2_ID,
        "college": COLLEGE_2,
        "gate_name": GATE_NAME_2,
        "source": CAMERA_SOURCE_2,
        "entry_line": ENTRY_LINE_2,
        "exit_line": EXIT_LINE_2,
        "enabled": True,
        "ws_path": "/ws/camera2",
    },
}


# ============================================================
# BACKWARD COMPATIBILITY EXPORTS (Legacy single-camera imports)
# ============================================================

CAMERA_SOURCE = CAMERA_SOURCE_1
CAMERA_ID = CAMERA_1_ID
GATE_NAME = GATE_NAME_1
ENTRY_LINE = ENTRY_LINE_1
EXIT_LINE = EXIT_LINE_1


# ============================================================
# AI / VEHICLE DETECTION
# ============================================================

YOLO_MODEL = "models/yolov8n.pt"
DETECTION_CONFIDENCE = 0.20
TRACKER_CONFIG = "bytetrack.yaml"


# ============================================================
# LINE CROSSING
# ============================================================

# Ignore very tiny point movements caused by tracker jitter.
LINE_CROSSING_MIN_MOVEMENT = 5

# Once a vehicle crosses its first line, its movement (ENTRY / EXIT) becomes locked.
FIRST_LINE_LOCK = True


# ============================================================
# OCR CONFIGURATION
# ============================================================

OCR_CONFIDENCE = 0.80

# OCR will be allowed only AFTER first virtual-line crossing.
OCR_AFTER_FIRST_LINE_ONLY = True

# Process OCR approximately once during this interval.
OCR_INTERVAL = 0.8

# Maximum number of OCR attempts for one tracked vehicle; 0 means unlimited.
MAX_OCR_FRAMES = 0

# Retry a transient college-bus verification database failure up to this many times.
DB_VERIFICATION_RETRIES = 3
DB_VERIFICATION_RETRY_INTERVAL = 2.0

# Number of repeated stable results required.
REQUIRED_STABLE_RESULTS = 2

# Post-crossing candidate frame sampling
BEST_FRAME_CANDIDATES = 4
BEST_FRAME_COLLECTION_SECONDS = 0.65


# ============================================================
# PLATE DETECTION
# ============================================================

PLATE_CONFIDENCE = 0.20
PLATE_IOU = 0.45
PLATE_IMAGE_SIZE = 640
MAX_PLATE_DETECTIONS = 5
MIN_PLATE_CONFIDENCE = 0.20


# ============================================================
# IDENTITY MATCHING
# ============================================================

# RapidFuzz minimum similarity score.
PLATE_FUZZY_THRESHOLD = 80
BUS_NUMBER_FUZZY_THRESHOLD = 80


# ============================================================
# TRACKING & BUFFERING
# ============================================================

# Brief missed detections do not expire a tracked vehicle.
TRACK_MISSING_TIMEOUT = 1.5

# Initial warmup frames.
WARMUP_FRAMES = 1

# Bounded frame buffer size to prevent video latency build-up
FRAME_QUEUE_SIZE = 2

# Background OCR job queue limit
# This is a per-camera bound; scheduling is round-robin across camera IDs.
OCR_QUEUE_SIZE = 30

# Number of timestamped frames retained for a bus before and during crossing.
PRE_CROSSING_FRAME_COUNT = 8

# Distance in pixels to line for preliminary zone detection
OCR_ZONE_DISTANCE_PIXELS = 450


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

DUPLICATE_COOLDOWN = 20  # seconds


# ============================================================
# CAMERA RETRY & WEBSOCKET
# ============================================================

CAMERA_RETRY_COUNT = 3
CAMERA_RETRY_DELAY = 2.0  # seconds between reconnect attempts

WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8000


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_NAME = os.getenv("POSTGRES_DB", "inoutx")


# ============================================================
# COLLECTION NAMES
# ============================================================

COLLEGE_BUS_COLLECTION = "college_buses"
BUS_LOG_COLLECTION = "bus_logs"
STAFF_VEHICLE_COLLECTION = "staff_vehicles"
STAFF_LOG_COLLECTION = "staff_logs"
UNKNOWN_VEHICLE_COLLECTION = "visitor_vehicles"
VEHICLE_STATE_COLLECTION = "vehicle_states"
GATE_EVENT_COLLECTION = "gate_events"


# ============================================================
# STORAGE & MEDIA
# ============================================================

CAPTURE_FOLDER = "captured_frames"
REPORT_FOLDER = "reports"
ALARM_SOUND = "alarm.wav"