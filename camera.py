# ============================================================
# IN/OUT X — DUAL-CAMERA PROCESSING ENGINE (KRCT & KRCE)
# ============================================================

import os
import cv2
import time
import queue
import asyncio
import threading
from collections import Counter, deque
from datetime import datetime
import numpy as np
import websockets

from config import (
    CAMERAS,
    CAMERA_1_ID,
    CAMERA_2_ID,
    COLLEGE_1,
    COLLEGE_2,
    GATE_NAME_1,
    GATE_NAME_2,
    CAMERA_SOURCE_1,
    CAMERA_SOURCE_2,
    ENTRY_LINE_1,
    EXIT_LINE_1,
    ENTRY_LINE_2,
    EXIT_LINE_2,
    LINE_CROSSING_MIN_MOVEMENT,
    OCR_AFTER_FIRST_LINE_ONLY,
    DUPLICATE_COOLDOWN,
    CAMERA_RETRY_COUNT,
    CAMERA_RETRY_DELAY,
    MIN_PLATE_CONFIDENCE,
    TRACK_MISSING_TIMEOUT,
    FRAME_QUEUE_SIZE,
    OCR_QUEUE_SIZE,
    OCR_INTERVAL,
    MAX_OCR_FRAMES,
    REQUIRED_STABLE_RESULTS,
    BEST_FRAME_CANDIDATES,
    BEST_FRAME_COLLECTION_SECONDS,
    OCR_ZONE_DISTANCE_PIXELS,
    WEBSOCKET_HOST,
    WEBSOCKET_PORT,
    mask_rtsp_url,
    validate_rtsp_url,
)

from tracker import VehicleTracker
from frame_capture import save_frame
from plate_detector import detect_plate
from ocr import read_plate, preload_ocr
from metadata import create_metadata
from notification import create_notification

from mongodb import (
    insert_bus_log,
    find_college_bus,
    verify_college_bus,
    find_best_matching_bus,
    find_bus_by_number,
    find_staff_vehicle,
    save_staff_vehicle,
    save_unknown_vehicle,
)
from config import reload_calibrated_lines
from line_calibration import (
    save_calibrated_lines,
    get_current_lines,
    prompt_edit_line,
    view_current_coordinates,
    reset_all_lines,
    apply_lines_to_processors,
)

# Vehicle class names (COCO)
VEHICLE_NAMES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK",
}


def format_camera_source(source: int | str) -> str:
    """Safely format and mask camera source for terminal logs and display."""
    if isinstance(source, int):
        return f"Camera Index {source}"
    if isinstance(source, str):
        if source.strip().isdigit():
            return f"Camera Index {int(source.strip())}"
        return mask_rtsp_url(source)
    return str(source)


# ============================================================
# LINE GEOMETRY HELPERS
# ============================================================

def _orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _line_to_points(line):
    if line is None:
        return None
    try:
        start, end = line
        return (
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
        )
    except Exception:
        return None


def line_touches_box(box, line):
    if box is None:
        return False
    points = _line_to_points(line)
    if points is None:
        return False
    try:
        x1, y1, x2, y2 = map(int, box)
    except Exception:
        return False

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    line_start, line_end = points
    box_edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]

    for edge_start, edge_end in box_edges:
        if segments_intersect(edge_start, edge_end, line_start, line_end):
            return True

    for px, py in (line_start, line_end):
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    return False


def box_center(box):
    if box is None:
        return None
    try:
        x1, y1, x2, y2 = map(int, box)
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    except Exception:
        return None


def distance_between_points(point_a, point_b):
    if point_a is None or point_b is None:
        return float("inf")
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return dx * dx + dy * dy


def point_to_segment_distance(point, line):
    if point is None:
        return float("inf")
    points = _line_to_points(line)
    if points is None:
        return float("inf")
    try:
        px, py = point
        x1, y1 = points[0]
        x2, y2 = points[1]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
    except Exception:
        return float("inf")


# ============================================================
# IMAGE & OCR UTILITIES
# ============================================================

def zoom_vehicle_crop(crop, scale=2.0):
    """Enlarge a vehicle crop by `scale` using cv2.resize and cv2.INTER_CUBIC.

    Safely handles empty, None, or invalid crops by returning the original crop or None.
    """
    if crop is None:
        return None
    try:
        if not hasattr(crop, "shape") or len(crop.shape) < 2:
            return crop
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0 or getattr(crop, "size", 0) == 0:
            return crop
        if scale <= 0 or scale == 1.0:
            return crop
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w <= 0 or new_h <= 0:
            return crop
        return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return crop


def vehicle_frame_quality(frame, box):
    """Estimate frame sharpness and size without expensive inference."""
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return -1.0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return -1.0
        crop_h, crop_w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        size_factor = min(1.5, (crop_w * crop_h) ** 0.5 / 300.0)
        return sharpness * max(0.5, size_factor)
    except Exception:
        return -1.0


def get_best_plate_crop(vehicle_crop, plate_results):
    if vehicle_crop is None or not plate_results:
        return None, None, 0.0
    try:
        vehicle_height, vehicle_width = vehicle_crop.shape[:2]
    except Exception:
        return None, None, 0.0

    best_box = None
    best_area = 0
    best_confidence = 0.0

    for detection in plate_results:
        if not isinstance(detection, dict):
            continue
        try:
            confidence = float(detection.get("confidence", 0.0))
            if confidence < MIN_PLATE_CONFIDENCE:
                continue
            raw_box = detection.get("box")
            if not raw_box or len(raw_box) != 4:
                continue
            px1, py1, px2, py2 = map(int, raw_box)
            px1 = max(0, min(px1, vehicle_width - 1))
            py1 = max(0, min(py1, vehicle_height - 1))
            px2 = max(0, min(px2, vehicle_width))
            py2 = max(0, min(py2, vehicle_height))
            if px2 <= px1 or py2 <= py1:
                continue
            area = (px2 - px1) * (py2 - py1)
            rank_score = confidence * (area ** 0.5)
            if rank_score > best_area:
                best_area = rank_score
                best_box = (px1, py1, px2, py2)
                best_confidence = confidence
        except Exception:
            pass

    if best_box is None:
        return None, None, 0.0

    px1, py1, px2, py2 = best_box
    pad_x = max(2, int((px2 - px1) * 0.12))
    pad_y = max(2, int((py2 - py1) * 0.25))
    crop_x1 = max(0, px1 - pad_x)
    crop_y1 = max(0, py1 - pad_y)
    crop_x2 = min(vehicle_width, px2 + pad_x)
    crop_y2 = min(vehicle_height, py2 + pad_y)

    plate_crop = vehicle_crop[crop_y1:crop_y2, crop_x1:crop_x2]
    if plate_crop.size == 0:
        return None, best_box, best_confidence
    return plate_crop, best_box, best_confidence


def clean_result(value):
    if value is None:
        return ""
    return str(value).strip().upper()


def get_most_common(values):
    if not values:
        return ""
    cleaned = [clean_result(v) for v in values if clean_result(v)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def is_stable(values):
    if not values:
        return False
    cleaned = [clean_result(v) for v in values if clean_result(v)]
    if not cleaned:
        return False
    _, count = Counter(cleaned).most_common(1)[0]
    return count >= REQUIRED_STABLE_RESULTS


def normalize_bus_number(value):
    if not value:
        return ""
    return "".join(char for char in str(value).strip().upper() if char.isalnum())


def save_detection_image(frame, track_id):
    try:
        return save_frame(frame, track_id)
    except Exception as e:
        print("Frame save error:", e)
        return ""


# ============================================================
# SHARED ASYNCHRONOUS OCR WORKER
# ============================================================

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class OCRJob:
    priority: int
    job_id: int
    payload: Any = field(compare=False)


ocr_queue = queue.PriorityQueue(maxsize=OCR_QUEUE_SIZE)
ocr_worker_running = False
ocr_worker_thread = None
ocr_job_counter = 0


def start_ocr_worker():
    global ocr_worker_running, ocr_worker_thread
    if ocr_worker_running:
        return
    ocr_worker_running = True
    ocr_worker_thread = threading.Thread(
        target=_ocr_worker_loop,
        daemon=True,
        name="SharedOCRWorker"
    )
    ocr_worker_thread.start()
    print("[OCR WORKER] Background OCR worker started.")


def stop_ocr_worker():
    global ocr_worker_running, ocr_worker_thread
    ocr_worker_running = False
    if ocr_worker_thread is not None:
        ocr_worker_thread.join(timeout=2.0)
        ocr_worker_thread = None
    print("[OCR WORKER] Background OCR worker stopped.")


def _ocr_worker_loop():
    while ocr_worker_running:
        try:
            job_item = ocr_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if job_item is None:
            continue

        payload = job_item.payload if isinstance(job_item, OCRJob) else job_item[2]
        camera_proc = payload["camera_proc"]
        track_id = payload["track_id"]
        crop = payload["crop"]
        vtype = payload["vehicle_type"]
        vbox = payload.get("vehicle_box")

        try:
            # 1. Plate detection
            plate_crop, plate_box, plate_conf = None, None, 0.0
            try:
                plate_results = detect_plate(crop)
                plate_crop, plate_box, plate_conf = get_best_plate_crop(crop, plate_results)
            except Exception as pe:
                print(f"[OCR] Plate detection error ({camera_proc.camera_id} Track {track_id}):", pe)

            # 2. Plate OCR
            plate_text = ""
            if plate_crop is not None:
                try:
                    plate_text = clean_result(read_plate(plate_crop))
                except Exception as oe:
                    print(f"[OCR] Plate OCR error ({camera_proc.camera_id} Track {track_id}):", oe)

            # 3. Deliver result back to the originating camera processor
            camera_proc.on_ocr_result(
                track_id=track_id,
                plate_text=plate_text,
                plate_box=plate_box,
                plate_conf=plate_conf,
                vehicle_type=vtype,
            )

        except Exception as e:
            print(f"[OCR] Error processing track {track_id} for {camera_proc.camera_id}:", e)
        finally:
            try:
                ocr_queue.task_done()
            except Exception:
                pass


# ============================================================
# CAMERA PROCESSOR CLASS (Independent worker per camera)
# ============================================================

class CameraProcessor:
    """
    Independent processing engine for one camera stream.
    Maintains its own:
      - RTSP capture loop & auto-reconnection
      - Latest-frame buffer
      - YOLO Tracker & ByteTrack state
      - Virtual entry & exit lines
      - Vehicle tracking & crossing state
      - Duplicate protection cooldowns
      - Camera metadata & status reporting
    """

    def __init__(
        self,
        camera_id: str,
        college: str,
        gate_name: str,
        source: str | int,
        entry_line: tuple,
        exit_line: tuple,
        ws_path: str = "/ws/camera1",
        enabled: bool = True,
        tracker: VehicleTracker | None = None,
    ):
        self.camera_id = camera_id
        self.college = college
        self.gate_name = gate_name
        self.source = source
        self.entry_line = entry_line
        self.exit_line = exit_line
        self.ws_path = ws_path
        self.enabled = enabled

        # Frame dimensions (dynamic, defaults to 1080p)
        self.frame_width = 1920
        self.frame_height = 1080

        # Status
        self.status = "INITIALIZING"
        self.status_message = "Starting worker..."
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_time = time.time()

        # Frames & Threading
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        self.frame_queue = deque(maxlen=FRAME_QUEUE_SIZE)
        self.capture_lock = threading.Lock()
        self.capture_ready = threading.Event()

        self.latest_annotated_frame = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()

        # Independent Tracker (Isolates ByteTrack state and ID space)
        self.tracker = tracker if tracker is not None else VehicleTracker()

        # State tracking
        self.vehicle_data = {}
        self.recent_plates = {}
        self.recent_bus_numbers = {}

        # OCR state per track (RLock ensures nested helper calls never deadlock)
        self.ocr_state = {}
        self.ocr_lock = threading.RLock()

        # Line calibration state
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_done = False

    # --------------------------------------------------------
    # CAPTURE WORKER
    # --------------------------------------------------------

    def start(self):
        if not self.enabled:
            self.status = "DISABLED"
            self.status_message = "Camera is disabled in configuration."
            print(f"[{self.camera_id}] Camera is disabled. Skipping worker.")
            return

        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
            name=f"Capture-{self.camera_id}"
        )
        self.capture_thread.start()

        self.process_thread = threading.Thread(
            target=self._process_worker,
            daemon=True,
            name=f"Process-{self.camera_id}"
        )
        self.process_thread.start()

        masked = format_camera_source(self.source)
        print(f"[{self.camera_id}] Started capture & processing threads for source: {masked}")

    def stop(self):
        self.running = False
        self.capture_ready.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        if self.process_thread is not None:
            self.process_thread.join(timeout=2.0)
            self.process_thread = None
        self.status = "STOPPED"
        print(f"[{self.camera_id}] Camera processor stopped.")

    def _open_stream(self):
        masked = format_camera_source(self.source)
        print(f"[{self.camera_id}] Connecting to stream: {masked}")

        cam_source = self.source
        if isinstance(cam_source, str) and cam_source.strip().isdigit():
            cam_source = int(cam_source.strip())

        for attempt in range(1, CAMERA_RETRY_COUNT + 1):
            if not self.running:
                return None

            self.status = "CONNECTING"
            self.status_message = f"Connecting (attempt {attempt}/{CAMERA_RETRY_COUNT})..."

            cap = cv2.VideoCapture(cam_source, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if cap.isOpened():
                self.status = "LIVE"
                self.status_message = "Connected and receiving stream."
                print(f"[{self.camera_id}] Successfully connected to stream.")
                time.sleep(0.3)
                return cap

            cap.release()
            print(f"[{self.camera_id}] Connection attempt {attempt} failed.")
            if attempt < CAMERA_RETRY_COUNT and self.running:
                time.sleep(CAMERA_RETRY_DELAY)

        self.status = "DISCONNECTED"
        self.status_message = "Stream unreachable. Will retry in background."
        return None

    def _capture_worker(self):
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None

                cap = self._open_stream()
                if cap is None:
                    if not self.running:
                        break
                    time.sleep(CAMERA_RETRY_DELAY)
                    continue

            ret, frame = cap.read()
            if ret and frame is not None:
                self.status = "LIVE"
                self.status_message = "Receiving live video frames."
                self.frame_width = frame.shape[1]
                self.frame_height = frame.shape[0]
                with self.capture_lock:
                    self.frame_queue.append(frame)
                self.capture_ready.set()
            else:
                print(f"[{self.camera_id}] Frame read failed or stream lost. Reconnecting...")
                self.status = "RECONNECTING"
                self.status_message = "Frame read failed. Reconnecting stream..."
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                with self.capture_lock:
                    self.frame_queue.clear()
                self.capture_ready.clear()
                time.sleep(CAMERA_RETRY_DELAY)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def get_latest_frame(self):
        with self.capture_lock:
            if not self.frame_queue:
                return None
            frame = self.frame_queue[-1]
            self.frame_queue.clear()
            return frame

    # --------------------------------------------------------
    # OCR STATE & SUBMISSION
    # --------------------------------------------------------

    def _ensure_ocr_state(self, track_id):
        with self.ocr_lock:
            if track_id not in self.ocr_state:
                self.ocr_state[track_id] = {
                    "plates": [],
                    "bus_numbers": [],
                    "plate_box": None,
                    "plate_confidence": 0.0,
                    "submitted": 0,
                    "completed": 0,
                    "pending": False,
                    "last_submit": 0.0,
                }
            return self.ocr_state[track_id]

    def _remove_ocr_state(self, track_id):
        with self.ocr_lock:
            self.ocr_state.pop(track_id, None)
    
    def submit_ocr_job(self, track_id, frame, vehicle_box, vehicle_type, priority=1):
        global ocr_job_counter
        try:
            x1, y1, x2, y2 = vehicle_box
            crop = frame[y1:y2, x1:x2].copy()
            if crop is None or crop.size == 0:
                return False

            crop = zoom_vehicle_crop(crop, scale=2.0)
            if crop is None or crop.size == 0:
                return False

            ocr_job_counter += 1
            payload = {
                "camera_proc": self,
                "track_id": track_id,
                "crop": crop,
                "vehicle_type": vehicle_type,
                "vehicle_box": vehicle_box,
            }
            job = OCRJob(priority=priority, job_id=ocr_job_counter, payload=payload)
            ocr_queue.put_nowait(job)

            state = self._ensure_ocr_state(track_id)
            with self.ocr_lock:
                state["submitted"] += 1
                state["pending"] = True
                state["last_submit"] = time.time()
            return True
        except queue.Full:
            print(f"[{self.camera_id}] OCR queue full. Skipping track {track_id}.")
            return False
        except Exception as e:
            print(f"[{self.camera_id}] Failed to submit OCR job:", e)
            return False

    def on_ocr_result(self, track_id, plate_text, plate_box, plate_conf, vehicle_type):
        with self.ocr_lock:
            state = self.ocr_state.get(track_id)
            if state is None:
                return
            if plate_text:
                state["plates"].append(plate_text)
                state["plates"] = state["plates"][-5:]
            if plate_box is not None:
                state["plate_box"] = plate_box
            state["plate_confidence"] = plate_conf
            state["completed"] += 1
            state["pending"] = False

        print(
            f"[{self.camera_id} OCR] Track {track_id} | Plate: {plate_text or 'NOT READ'} | Conf: {plate_conf:.2f}"
        )

    # --------------------------------------------------------
    # LINE CROSSING CHECK
    # --------------------------------------------------------

    def check_first_touched_line(self, previous_box, current_box):
        if current_box is None:
            return None

        touched = []
        exit_points = _line_to_points(self.exit_line)
        entry_points = _line_to_points(self.entry_line)

        if exit_points is not None and line_touches_box(current_box, exit_points):
            touched.append("EXIT")
        if entry_points is not None and line_touches_box(current_box, entry_points):
            touched.append("ENTRY")

        if not touched:
            return None
        if len(touched) == 1:
            return touched[0]

        # If touches both, disambiguate using previous distance
        if previous_box is None:
            return "EXIT"
        prev_center = box_center(previous_box)
        if prev_center is None:
            return "EXIT"

        exit_mid = ((exit_points[0][0] + exit_points[1][0]) // 2, (exit_points[0][1] + exit_points[1][1]) // 2) if exit_points else None
        entry_mid = ((entry_points[0][0] + entry_points[1][0]) // 2, (entry_points[0][1] + entry_points[1][1]) // 2) if entry_points else None

        d_exit = distance_between_points(prev_center, exit_mid)
        d_entry = distance_between_points(prev_center, entry_mid)
        return "EXIT" if d_exit <= d_entry else "ENTRY"

    # --------------------------------------------------------
    # DATABASE & NOTIFICATIONS
    # --------------------------------------------------------

    def is_duplicate(self, plate, current_time):
        plate = clean_result(plate)
        if not plate:
            return False
        last_seen = self.recent_plates.get(plate)
        if last_seen is None:
            return False
        return (current_time - last_seen) < DUPLICATE_COOLDOWN

    def remember_plate(self, plate, current_time):
        plate = clean_result(plate)
        if plate:
            self.recent_plates[plate] = current_time

    def classify_and_save(self, frame, track_id, data, current_time):
        vtype = data.get("vehicle_type", "VEHICLE")
        movement = data.get("movement_status") or "ENTRY"
        crossing_time = data.get("crossing_time") or datetime.now()

        with self.ocr_lock:
            state = self.ocr_state.get(track_id, {})
            plates = list(state.get("plates", []))

        stable_plate = get_most_common(plates)
        plate_is_stable = is_stable(plates)

        if not plate_is_stable and vtype != "BUS":
            return False

        # STRICT COLLEGE BUS VERIFICATION
        bus_data = None
        if stable_plate or plates:
            try:
                bus_data = verify_college_bus(stable_plate, plausible_candidates=plates)
            except Exception as e:
                print(f"[{self.camera_id}] DB bus lookup error:", e)

        if bus_data is not None:
            # Verified College Bus Record
            atlas_bus_no = bus_data.get("bus_no", bus_data.get("bus_number", ""))
            atlas_plate = bus_data.get("plate", stable_plate)

            if self.is_duplicate(atlas_plate, current_time):
                print(f"[{self.camera_id}] Duplicate college bus ignored: {atlas_plate}")
                return True

            img_path = save_detection_image(frame, track_id)
            meta = {
                "camera": f"{self.college} Gate Camera",
                "camera_id": self.camera_id,
                "college": self.college,
                "gate": self.gate_name,
                "gate_name": self.gate_name,
                "bus_id": bus_data.get("bus_id", ""),
                "bus_no": atlas_bus_no,
                "bus_number": atlas_bus_no,
                "plate": atlas_plate,
                "ocr_plate": stable_plate or atlas_plate,
                "driver": bus_data.get("driver", ""),
                "route": bus_data.get("route", ""),
                "registered_status": bus_data.get("status", "ACTIVE"),
                "movement": movement,
                "direction": "IN" if movement == "ENTRY" else "OUT",
                "status": "ARRIVED" if movement == "ENTRY" else "EXITED",
                "entry_time": crossing_time if movement == "ENTRY" else None,
                "exit_time": crossing_time if movement == "EXIT" else None,
                "classification": "COLLEGE_BUS",
                "vehicle_type": "BUS",
                "image": img_path,
                "date": datetime.now().strftime("%d-%m-%Y"),
                "time": datetime.now().strftime("%I:%M:%S %p"),
                "timestamp": crossing_time,
                "detected_at": datetime.now(),
            }

            try:
                insert_bus_log(meta)
                print(f"[{self.camera_id}] {self.college} COLLEGE BUS {movement} SAVED: Bus #{atlas_bus_no} ({atlas_plate})")
            except Exception as e:
                print(f"[{self.camera_id}] DB error saving college bus:", e)

            self.remember_plate(atlas_plate, current_time)
            create_notification(
                title=f"{self.college} College Bus Verified",
                message=f"Bus #{atlas_bus_no} ({atlas_plate}) {movement.lower()}ed {self.gate_name}.",
                notification_type="success",
                college=self.college,
                camera_id=self.camera_id
            )
            return True

        # Unregistered bus or unverified plate: NEVER log event and NEVER send notification
        with self.ocr_lock:
            submitted = self.ocr_state.get(track_id, {}).get("submitted", 0)
        if submitted >= MAX_OCR_FRAMES and plates:
            print(f"[{self.camera_id}] Bus ID {track_id} (plates={plates}) is NOT a registered college bus. Skipped.")

        return False

    # --------------------------------------------------------
    # DRAWING & VISUALIZATION
    # --------------------------------------------------------

    def draw_lines(self, frame):
        exit_points = _line_to_points(self.exit_line)
        if exit_points is not None:
            # Vibrant Orange for EXIT line
            cv2.line(frame, exit_points[0], exit_points[1], (0, 140, 255), 3)
            # Clear endpoint markers
            cv2.circle(frame, exit_points[0], 7, (255, 255, 255), -1)
            cv2.circle(frame, exit_points[0], 5, (0, 140, 255), -1)
            cv2.circle(frame, exit_points[1], 7, (255, 255, 255), -1)
            cv2.circle(frame, exit_points[1], 5, (0, 140, 255), -1)
            label_x = min(exit_points[0][0], exit_points[1][0]) + 10
            label_y = max(25, min(exit_points[0][1], exit_points[1][1]) - 10)
            cv2.putText(
                frame,
                f"{self.college} EXIT",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 140, 255),
                2,
            )

        entry_points = _line_to_points(self.entry_line)
        if entry_points is not None:
            # Vibrant Yellow for ENTRY line
            cv2.line(frame, entry_points[0], entry_points[1], (0, 255, 255), 3)
            # Clear endpoint markers
            cv2.circle(frame, entry_points[0], 7, (255, 255, 255), -1)
            cv2.circle(frame, entry_points[0], 5, (0, 255, 255), -1)
            cv2.circle(frame, entry_points[1], 7, (255, 255, 255), -1)
            cv2.circle(frame, entry_points[1], 5, (0, 255, 255), -1)
            label_x = min(entry_points[0][0], entry_points[1][0]) + 10
            label_y = max(25, min(entry_points[0][1], entry_points[1][1]) - 10)
            cv2.putText(
                frame,
                f"{self.college} ENTRY",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
            )

    def draw_status_header(self, frame):
        """Draw an informative top banner showing camera identity, college, and stream status."""
        h, w = frame.shape[:2]
        banner_h = 35
        cv2.rectangle(frame, (0, 0), (w, banner_h), (20, 20, 25), -1)

        status_color = (0, 255, 128) if self.status == "LIVE" else (0, 165, 255)
        if "WARNING" in self.status or "DISCONNECTED" in self.status:
            status_color = (0, 0, 255)

        cv2.circle(frame, (15, 18), 6, status_color, -1)
        text = f"{self.college} | {self.camera_id} | {self.gate_name} | {self.status}"
        cv2.putText(frame, text, (30, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # --------------------------------------------------------
    # MAIN PROCESSING LOOP
    # --------------------------------------------------------

    def _process_worker(self):
        while self.running:
            raw_frame = self.get_latest_frame()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            self.frame_width = raw_frame.shape[1]
            self.frame_height = raw_frame.shape[0]

            current_time = time.time()
            self.frame_count += 1
            dt = current_time - self.last_frame_time
            if dt >= 1.0:
                self.fps = self.frame_count / dt
                self.frame_count = 0
                self.last_frame_time = current_time

            # 1. Detect & Track with this camera's independent tracker
            try:
                results = self.tracker.track(raw_frame)
            except Exception as te:
                print(f"[{self.camera_id}] Tracking error:", te)
                results = []

            # 2. Annotate base
            annotated = raw_frame.copy()
            if results:
                try:
                    annotated = results[0].plot()
                except Exception:
                    annotated = raw_frame.copy()

            self.draw_lines(annotated)
            self.draw_status_header(annotated)

            current_track_ids = set()

            # 3. Process track boxes
            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                for box in result.boxes:
                    if box.id is None or box.cls is None or box.xyxy is None:
                        continue
                    try:
                        b_id = box.id
                        b_cls = box.cls
                        b_xyxy = box.xyxy
                        if b_id is None or b_cls is None or b_xyxy is None:
                            continue
                        track_id = int(b_id[0])
                        class_id = int(b_cls[0])
                        current_box = tuple(map(int, b_xyxy[0].tolist()))
                    except Exception:
                        continue

                    # STRICT VEHICLE FILTER: Only process YOLO bus class (ID 5)
                    # Cars (2), motorcycles (3), trucks (7), and others are strictly ignored
                    if class_id != 5:
                        continue

                    current_track_ids.add(track_id)
                    vtype = VEHICLE_NAMES.get(class_id, "VEHICLE")

                    # Initialize track state if new
                    if track_id not in self.vehicle_data:
                        self.vehicle_data[track_id] = {
                            "last_seen": current_time,
                            "vehicle_type": vtype,
                            "previous_box": None,
                            "current_box": current_box,
                            "first_crossed_line": None,
                            "movement_status": None,
                            "crossing_time": None,
                            "identity_enabled": False,
                            "processed": False,
                            "candidates": [],
                            "collection_start": None,
                            "last_frame": raw_frame.copy(),
                        }
                        self._ensure_ocr_state(track_id)

                    tdata = self.vehicle_data[track_id]
                    tdata["last_seen"] = current_time
                    tdata["vehicle_type"] = vtype
                    tdata["last_frame"] = raw_frame.copy()
                    tdata["previous_box"] = tdata.get("current_box")
                    tdata["current_box"] = current_box

                    bx1, by1, bx2, by2 = current_box

                    # Check first line touch
                    if tdata["first_crossed_line"] is None:
                        touched = self.check_first_touched_line(tdata["previous_box"], current_box)
                        if touched is not None:
                            tdata["first_crossed_line"] = touched
                            tdata["movement_status"] = touched
                            tdata["crossing_time"] = datetime.now()
                            tdata["identity_enabled"] = True
                            print(f"[{self.camera_id}] Line touch: {vtype} ID {track_id} -> {touched}")

                    # Render movement label
                    move_lbl = tdata.get("movement_status") or "WAITING FOR LINE"
                    cv2.putText(
                        annotated,
                        move_lbl,
                        (bx1, max(35, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                    if tdata["processed"]:
                        cv2.putText(
                            annotated,
                            "PROCESSED",
                            (bx1, min(annotated.shape[0] - 10, by2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (0, 255, 0),
                            2,
                        )
                        continue

                    # Pre-crossing: Show BUS DETECTED, do NOT run OCR
                    if not tdata["identity_enabled"]:
                        cv2.putText(
                            annotated,
                            "BUS DETECTED - WAITING FOR LINE" if vtype == "BUS" else "DETECTED - WAITING FOR LINE",
                            (bx1, min(annotated.shape[0] - 10, by2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (255, 255, 0),
                            2,
                        )
                        continue

                    # Post-crossing: Collect candidates and trigger OCR
                    with self.ocr_lock:
                        ostate = self._ensure_ocr_state(track_id)
                        attempts = ostate["submitted"]
                        pending = ostate["pending"]
                        last_sub = ostate["last_submit"]

                    if attempts < MAX_OCR_FRAMES and not pending:
                        cands = tdata.setdefault("candidates", [])
                        col_start = tdata.get("collection_start")
                        if col_start is None:
                            tdata["collection_start"] = current_time
                            cands.clear()

                        if len(cands) < BEST_FRAME_CANDIDATES:
                            score = vehicle_frame_quality(raw_frame, current_box)
                            if score > 0:
                                cands.append((score, raw_frame.copy(), current_box))

                        col_start = tdata.get("collection_start", current_time)
                        if (
                            len(cands) >= BEST_FRAME_CANDIDATES
                            or (current_time - col_start) >= BEST_FRAME_COLLECTION_SECONDS
                        ):
                            if cands and (current_time - last_sub) >= OCR_INTERVAL:
                                best_s, best_f, best_b = max(cands, key=lambda it: it[0])
                                submitted = self.submit_ocr_job(track_id, best_f, best_b, vtype, priority=1)
                                if submitted:
                                    print(f"[{self.camera_id}] Post-crossing OCR queued: ID {track_id} (quality={best_s:.1f})")
                                cands.clear()
                                tdata["collection_start"] = current_time

                    # Try classification
                    saved = self.classify_and_save(raw_frame, track_id, tdata, current_time)
                    if saved:
                        tdata["processed"] = True

            # 4. Clean up missing tracks & cooldown memory
            for tid, tdata in list(self.vehicle_data.items()):
                if tid not in current_track_ids:
                    missing = current_time - tdata.get("last_seen", current_time)
                    if missing > TRACK_MISSING_TIMEOUT:
                        if tdata.get("identity_enabled") and not tdata.get("processed"):
                            # Final chance save
                            self.classify_and_save(tdata.get("last_frame", raw_frame), tid, tdata, current_time)
                        if missing > 5.0:
                            self.vehicle_data.pop(tid, None)
                            self._remove_ocr_state(tid)

            # 5. Clean duplicate plates cache
            for p, ts in list(self.recent_plates.items()):
                if current_time - ts > DUPLICATE_COOLDOWN:
                    self.recent_plates.pop(p, None)

            # 6. Save latest annotated frame for streaming & display
            with self.frame_lock:
                self.latest_annotated_frame = annotated.copy()
                self.latest_raw_frame = raw_frame.copy()


# ============================================================
# MULTI-CAMERA MANAGER
# ============================================================

camera_processors: dict[str, CameraProcessor] = {}


def initialize_cameras():
    global camera_processors
    camera_processors.clear()

    # Reload latest line coordinates from persistent storage
    reload_calibrated_lines()

    # Camera 1 — KRCT
    camera_processors[CAMERA_1_ID] = CameraProcessor(
        camera_id=CAMERA_1_ID,
        college=COLLEGE_1,
        gate_name=GATE_NAME_1,
        source=CAMERA_SOURCE_1,
        entry_line=ENTRY_LINE_1,
        exit_line=EXIT_LINE_1,
        ws_path="/ws/camera1",
        enabled=True,
    )

    # Camera 2 — KRCE
    camera_processors[CAMERA_2_ID] = CameraProcessor(
        camera_id=CAMERA_2_ID,
        college=COLLEGE_2,
        gate_name=GATE_NAME_2,
        source=CAMERA_SOURCE_2,
        entry_line=ENTRY_LINE_2,
        exit_line=EXIT_LINE_2,
        ws_path="/ws/camera2",
        enabled=True,
    )


def get_processor(camera_id: str) -> CameraProcessor | None:
    return camera_processors.get(camera_id)


# ============================================================
# DUAL-CAMERA WEBSOCKET SERVER
# ============================================================

def create_disconnected_placeholder(college: str, camera_id: str, message: str):
    """Generate a clean branded placeholder frame when a camera is disconnected."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (24, 24, 28)

    # Top banner
    cv2.rectangle(img, (0, 0), (640, 40), (15, 15, 18), -1)
    cv2.putText(
        img,
        f"{college} — {camera_id}",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # Center status
    cv2.circle(img, (320, 200), 30, (0, 0, 180), -1)
    cv2.putText(
        img,
        "!",
        (314, 212),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    cv2.putText(
        img,
        "STREAM DISCONNECTED",
        (190, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 100, 255),
        2,
    )

    cv2.putText(
        img,
        message[:45],
        (100, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
    )
    return img


async def camera_websocket_handler(websocket, path: str | None = None):
    """
    Multi-path WebSocket endpoint:
      - /ws/camera1 -> Camera 1 (KRCT)
      - /ws/camera2 -> Camera 2 (KRCE)
      - /ws/camera  -> Alias for Camera 1 (backward compatible)
    """
    if path is None:
        request = getattr(websocket, "request", None)
        if request is not None and hasattr(request, "path"):
            path = str(request.path)
        else:
            path = str(getattr(websocket, "path", ""))

    path = (path or "").rstrip("/")
    if path in ("/ws/camera1", "/ws/camera", ""):
        target_id = CAMERA_1_ID
    elif path == "/ws/camera2":
        target_id = CAMERA_2_ID
    else:
        target_id = CAMERA_1_ID

    proc = get_processor(target_id)
    if proc is None:
        await websocket.close()
        return

    try:
        while True:
            frame = None
            with proc.frame_lock:
                if proc.latest_annotated_frame is not None:
                    frame = proc.latest_annotated_frame.copy()

            # If no frame yet or stream disconnected, stream placeholder
            if frame is None or proc.status != "LIVE":
                frame = create_disconnected_placeholder(
                    college=proc.college,
                    camera_id=proc.camera_id,
                    message=proc.status_message,
                )

            h, w = frame.shape[:2]
            max_w = 960
            if w > max_w:
                new_h = int(h * max_w / w)
                frame = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if success:
                await websocket.send(buffer.tobytes())

            await asyncio.sleep(0.1)  # ~10 fps

    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"WebSocket error ({target_id}):", e)


try:
    from websockets.http11 import Response as WsHttpResponse
    from websockets.datastructures import Headers as WsHttpHeaders
except Exception:
    WsHttpResponse = None
    WsHttpHeaders = None


CAMERA_PAGE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IN/OUT X - Live Camera Monitor</title>
    <link rel="icon" type="image/png" href="https://inoutx.zynix.us/assets/images/logo.PNG">
    <style>
        :root {
            --bg-main: #0a0e17;
            --bg-card: #111827;
            --bg-card-hover: #1e293b;
            --border-color: #1f2937;
            --border-glow: rgba(16, 185, 129, 0.3);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-green: #10b981;
            --accent-green-bg: rgba(16, 185, 129, 0.15);
            --accent-blue: #06b6d4;
            --accent-blue-bg: rgba(6, 182, 212, 0.15);
            --accent-indigo: #6366f1;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: inherit;
        }

        .brand-badge {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 18px;
            color: #ffffff;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        }

        .brand-title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-sub {
            font-size: 11px;
            color: var(--accent-green);
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .tabs {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(15, 23, 42, 0.6);
            padding: 4px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab-btn.active {
            color: #ffffff;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 182, 212, 0.25) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
        }

        .actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .btn-link {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 8px;
            text-decoration: none;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .btn-link:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.2);
        }

        main {
            flex: 1;
            padding: 24px;
            max-width: 1600px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .camera-grid {
            display: grid;
            gap: 20px;
            width: 100%;
        }

        .camera-grid.dual-view {
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
        }

        .camera-grid.single-view {
            grid-template-columns: 1fr;
            max-width: 1100px;
            margin: 0 auto;
        }

        .camera-card {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .camera-card:hover {
            border-color: rgba(16, 185, 129, 0.3);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(16, 185, 129, 0.1);
        }

        .card-header {
            padding: 14px 18px;
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .cam-meta {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .cam-name {
            font-size: 15px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.01em;
        }

        .cam-college {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--accent-green-bg);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent-green);
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px var(--accent-green); }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .video-container {
            position: relative;
            background: #000000;
            width: 100%;
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .stream-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .overlay-status {
            position: absolute;
            top: 12px;
            left: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 10;
        }

        .overlay-fps {
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(6px);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-family: monospace;
            color: #94a3b8;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .overlay-controls {
            position: absolute;
            top: 12px;
            right: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 10;
        }

        .icon-btn {
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #ffffff;
            width: 32px;
            height: 32px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .icon-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
            transform: scale(1.05);
        }

        .connecting-overlay {
            position: absolute;
            inset: 0;
            background: rgba(10, 14, 23, 0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            color: var(--text-secondary);
            font-size: 13px;
            z-index: 5;
            transition: opacity 0.3s ease;
        }

        .spinner {
            width: 36px;
            height: 36px;
            border: 3px solid rgba(16, 185, 129, 0.2);
            border-top-color: var(--accent-green);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .card-footer {
            padding: 10px 18px;
            background: rgba(15, 23, 42, 0.4);
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-muted);
        }

        .endpoint-pill {
            font-family: monospace;
            background: rgba(255, 255, 255, 0.04);
            padding: 2px 8px;
            border-radius: 4px;
            color: #cbd5e1;
            font-size: 11px;
        }

        @media (max-width: 768px) {
            .camera-grid.dual-view {
                grid-template-columns: 1fr;
            }
            header {
                padding: 10px 14px;
            }
            main {
                padding: 14px;
            }
            .tab-btn {
                padding: 6px 12px;
                font-size: 12px;
            }
        }
    </style>
</head>
<body data-view="__INITIAL_VIEW__">
    <header>
        <a href="https://inoutx.zynix.us" class="brand">
            <div class="brand-badge">X</div>
            <div>
                <div class="brand-title">IN/OUT X</div>
                <div class="brand-sub">Dual-Camera AI Engine</div>
            </div>
        </a>

        <div class="tabs">
            <button class="tab-btn" id="tab-cam1" onclick="switchView('camera1')">
                <span>KRCT Gate 01</span>
            </button>
            <button class="tab-btn" id="tab-cam2" onclick="switchView('camera2')">
                <span>KRCE Gate 01</span>
            </button>
            <button class="tab-btn" id="tab-dual" onclick="switchView('dual')">
                <span>Dual Split-View</span>
            </button>
        </div>

        <div class="actions">
            <a href="https://inoutx.zynix.us" class="btn-link">
                <span>← Dashboard</span>
            </a>
        </div>
    </header>

    <main>
        <div id="camera-grid" class="camera-grid dual-view">
            <!-- Camera 1 Card -->
            <div class="camera-card" id="card-cam1">
                <div class="card-header">
                    <div class="cam-meta">
                        <div>
                            <div class="cam-name">KRCT Gate 01</div>
                            <div class="cam-college">K. Ramakrishnan College of Technology</div>
                        </div>
                    </div>
                    <div class="badge-live" id="status-badge-1">
                        <span class="pulse-dot"></span>
                        <span id="status-text-1">CONNECTING</span>
                    </div>
                </div>
                <div class="video-container" id="container-cam1">
                    <img class="stream-img" id="stream-img-1" alt="KRCT Camera Live Stream">
                    <div class="overlay-status">
                        <span class="overlay-fps" id="fps-1">0 FPS</span>
                    </div>
                    <div class="overlay-controls">
                        <button class="icon-btn" title="Fullscreen" onclick="toggleFullscreen('container-cam1')">⛶</button>
                    </div>
                    <div class="connecting-overlay" id="loading-1">
                        <div class="spinner"></div>
                        <div>Connecting to KRCT Camera Stream...</div>
                    </div>
                </div>
                <div class="card-footer">
                    <span>Endpoint: <span class="endpoint-pill">/ws/camera1</span></span>
                    <span id="frames-count-1">0 frames</span>
                </div>
            </div>

            <!-- Camera 2 Card -->
            <div class="camera-card" id="card-cam2">
                <div class="card-header">
                    <div class="cam-meta">
                        <div>
                            <div class="cam-name">KRCE Gate 01</div>
                            <div class="cam-college">K. Ramakrishnan College of Engineering</div>
                        </div>
                    </div>
                    <div class="badge-live" id="status-badge-2">
                        <span class="pulse-dot"></span>
                        <span id="status-text-2">CONNECTING</span>
                    </div>
                </div>
                <div class="video-container" id="container-cam2">
                    <img class="stream-img" id="stream-img-2" alt="KRCE Camera Live Stream">
                    <div class="overlay-status">
                        <span class="overlay-fps" id="fps-2">0 FPS</span>
                    </div>
                    <div class="overlay-controls">
                        <button class="icon-btn" title="Fullscreen" onclick="toggleFullscreen('container-cam2')">⛶</button>
                    </div>
                    <div class="connecting-overlay" id="loading-2">
                        <div class="spinner"></div>
                        <div>Connecting to KRCE Camera Stream...</div>
                    </div>
                </div>
                <div class="card-footer">
                    <span>Endpoint: <span class="endpoint-pill">/ws/camera2</span></span>
                    <span id="frames-count-2">0 frames</span>
                </div>
            </div>
        </div>
    </main>

    <script>
        const initialView = document.body.dataset.view || "dual";
        let currentView = initialView;
        let socket1 = null;
        let socket2 = null;
        let fpsCounter1 = 0, fpsCounter2 = 0;
        let totalFrames1 = 0, totalFrames2 = 0;

        function getWsUrl(cameraPath) {
            const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
            return `${proto}//${window.location.host}${cameraPath}`;
        }

        setInterval(() => {
            const el1 = document.getElementById("fps-1");
            const el2 = document.getElementById("fps-2");
            if (el1) el1.textContent = `${fpsCounter1} FPS`;
            if (el2) el2.textContent = `${fpsCounter2} FPS`;
            fpsCounter1 = 0;
            fpsCounter2 = 0;
        }, 1000);

        function connectCam1() {
            if (socket1 && (socket1.readyState === WebSocket.OPEN || socket1.readyState === WebSocket.CONNECTING)) {
                return;
            }
            const img = document.getElementById("stream-img-1");
            const statusText = document.getElementById("status-text-1");
            const statusBadge = document.getElementById("status-badge-1");
            const loading = document.getElementById("loading-1");
            const framesCount = document.getElementById("frames-count-1");
            const url = getWsUrl("/ws/camera1");

            try {
                socket1 = new WebSocket(url);
                socket1.binaryType = "blob";
            } catch (e) {
                if (statusText) statusText.textContent = "FAILED";
                setTimeout(connectCam1, 3000);
                return;
            }

            socket1.onopen = () => {
                if (statusText) statusText.textContent = "LIVE";
                if (statusBadge) statusBadge.style.display = "inline-flex";
                if (loading) loading.style.display = "none";
            };

            socket1.onmessage = (event) => {
                if (event.data instanceof Blob && img) {
                    const blobUrl = URL.createObjectURL(event.data);
                    const old = img.dataset.url;
                    img.src = blobUrl;
                    img.dataset.url = blobUrl;
                    if (old) URL.revokeObjectURL(old);
                    fpsCounter1++;
                    totalFrames1++;
                    if (framesCount) framesCount.textContent = `${totalFrames1} frames`;
                    if (loading && loading.style.display !== "none") loading.style.display = "none";
                }
            };

            socket1.onerror = () => {
                if (statusText) statusText.textContent = "ERROR";
            };

            socket1.onclose = () => {
                if (statusText) statusText.textContent = "RECONNECTING";
                if (loading) loading.style.display = "flex";
                setTimeout(connectCam1, 2000);
            };
        }

        function connectCam2() {
            if (socket2 && (socket2.readyState === WebSocket.OPEN || socket2.readyState === WebSocket.CONNECTING)) {
                return;
            }
            const img = document.getElementById("stream-img-2");
            const statusText = document.getElementById("status-text-2");
            const statusBadge = document.getElementById("status-badge-2");
            const loading = document.getElementById("loading-2");
            const framesCount = document.getElementById("frames-count-2");
            const url = getWsUrl("/ws/camera2");

            try {
                socket2 = new WebSocket(url);
                socket2.binaryType = "blob";
            } catch (e) {
                if (statusText) statusText.textContent = "FAILED";
                setTimeout(connectCam2, 3000);
                return;
            }

            socket2.onopen = () => {
                if (statusText) statusText.textContent = "LIVE";
                if (statusBadge) statusBadge.style.display = "inline-flex";
                if (loading) loading.style.display = "none";
            };

            socket2.onmessage = (event) => {
                if (event.data instanceof Blob && img) {
                    const blobUrl = URL.createObjectURL(event.data);
                    const old = img.dataset.url;
                    img.src = blobUrl;
                    img.dataset.url = blobUrl;
                    if (old) URL.revokeObjectURL(old);
                    fpsCounter2++;
                    totalFrames2++;
                    if (framesCount) framesCount.textContent = `${totalFrames2} frames`;
                    if (loading && loading.style.display !== "none") loading.style.display = "none";
                }
            };

            socket2.onerror = () => {
                if (statusText) statusText.textContent = "ERROR";
            };

            socket2.onclose = () => {
                if (statusText) statusText.textContent = "RECONNECTING";
                if (loading) loading.style.display = "flex";
                setTimeout(connectCam2, 2000);
            };
        }

        function switchView(view) {
            currentView = view;
            const grid = document.getElementById("camera-grid");
            const card1 = document.getElementById("card-cam1");
            const card2 = document.getElementById("card-cam2");
            const tab1 = document.getElementById("tab-cam1");
            const tab2 = document.getElementById("tab-cam2");
            const tabDual = document.getElementById("tab-dual");

            tab1.classList.toggle("active", view === "camera1");
            tab2.classList.toggle("active", view === "camera2");
            tabDual.classList.toggle("active", view === "dual");

            if (view === "camera1") {
                grid.className = "camera-grid single-view";
                card1.style.display = "flex";
                card2.style.display = "none";
                connectCam1();
            } else if (view === "camera2") {
                grid.className = "camera-grid single-view";
                card1.style.display = "none";
                card2.style.display = "flex";
                connectCam2();
            } else {
                grid.className = "camera-grid dual-view";
                card1.style.display = "flex";
                card2.style.display = "flex";
                connectCam1();
                connectCam2();
            }
        }

        function toggleFullscreen(elementId) {
            const el = document.getElementById(elementId);
            if (!el) return;
            if (!document.fullscreenElement) {
                if (el.requestFullscreen) el.requestFullscreen();
                else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
            } else {
                if (document.exitFullscreen) document.exitFullscreen();
            }
        }

        switchView(initialView);
    </script>
</body>
</html>"""


def camera_http_handler(connection, request):
    """
    Handles standard browser HTTP GET requests to the camera server.
    If the client requests a WebSocket upgrade, returns None to allow the handshake.
    Otherwise, serves an interactive HTML live video player for KRCT / KRCE.
    """
    try:
        upgrade = (request.headers.get("upgrade") or "").lower()
        if upgrade == "websocket":
            return None

        if WsHttpResponse is None or WsHttpHeaders is None:
            return None

        path = (request.path or "").rstrip("/")
        if path in ("/ws/camera2", "/camera2"):
            initial_view = "camera2"
        elif path in ("/ws/camera1", "/camera1"):
            initial_view = "camera1"
        else:
            initial_view = "dual"

        html = CAMERA_PAGE_HTML_TEMPLATE.replace("__INITIAL_VIEW__", initial_view).encode("utf-8")
        headers = WsHttpHeaders([
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(html))),
            ("Cache-Control", "no-cache, no-store, must-revalidate"),
            ("Connection", "close"),
        ])
        return WsHttpResponse(200, "OK", headers, html)
    except Exception as e:
        print("[HTTP REQUEST ERROR]", e)
        return None


def run_websocket_server():
    async def start_server():
        try:
            server = await websockets.serve(
                camera_websocket_handler,  # type: ignore[arg-type]
                WEBSOCKET_HOST,
                WEBSOCKET_PORT,
                process_request=camera_http_handler,
            )
            print(f"[WEBSOCKET] Dual-Camera WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
            print(f"  - Camera 1 (KRCT): ws://localhost:{WEBSOCKET_PORT}/ws/camera1")
            print(f"  - Camera 2 (KRCE): ws://localhost:{WEBSOCKET_PORT}/ws/camera2")
            print(f"  - Web UI Monitor : http://localhost:{WEBSOCKET_PORT}/")
            await server.wait_closed()
        except OSError as err:
            print(f"[WEBSOCKET ERROR] Could not bind to ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}: {err}")
        except Exception as err:
            print(f"[WEBSOCKET ERROR] Server encountered an error: {err}")

    try:
        asyncio.run(start_server())
    except Exception as e:
        print(f"[WEBSOCKET] Server thread terminated: {e}")


# ============================================================
# INTERACTIVE MOUSE-BASED LINE CALIBRATION IN CAMERA PREVIEW
# ============================================================

from dataclasses import dataclass

WINDOW_NAME = "IN/OUT X - Dual Camera Live Monitor & Line Calibration"


@dataclass(frozen=True)
class HudButton:
    id: str
    key: str
    title: str
    desc: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def label(self) -> str:
        return self.title

    def __getitem__(self, item: str) -> Any:
        if item == "box":
            return (self.x1, self.y1, self.x2, self.y2)
        if item == "label":
            return self.title
        return getattr(self, item)


# Operational Modes: "CALIBRATION" or "DETECTION"
# Starts in DETECTION mode by default. Operator presses 'R' to enter calibration.
active_app_mode: str = "DETECTION"

# Currently selected line for mouse editing:
# "ENTRY_LINE_1", "EXIT_LINE_1", "ENTRY_LINE_2", "EXIT_LINE_2"
selected_calib_line: str = "ENTRY_LINE_1"

# Pending click points for currently selected line: list of ((disp_x, disp_y), (cam_x, cam_y))
calib_click_points: list[tuple[tuple[int, int], tuple[int, int]]] = []

# Current mouse hover coordinates
mouse_cursor_pos: tuple[int, int] | None = None

# Status message shown on the preview HUD
hud_status_text: str = "Detection Active: College bus monitoring running. Press [R] for Line Calibration."
hud_status_time: float = time.time()

LINE_NAME_MAP: dict[str, str] = {
    "ENTRY_LINE_1": "Entry Line 1 (KRCT)",
    "EXIT_LINE_1": "Exit Line 1 (KRCT)",
    "ENTRY_LINE_2": "Entry Line 2 (KRCE)",
    "EXIT_LINE_2": "Exit Line 2 (KRCE)",
}

HUD_BUTTONS: list[HudButton] = [
    HudButton("ENTRY_LINE_1", "1", "1. Entry 1", "KRCT Entry", 15, 515, 175, 555),
    HudButton("EXIT_LINE_1",  "2", "2. Exit 1",  "KRCT Exit",  185, 515, 345, 555),
    HudButton("ENTRY_LINE_2", "3", "3. Entry 2", "KRCE Entry", 355, 515, 515, 555),
    HudButton("EXIT_LINE_2",  "4", "4. Exit 2",  "KRCE Exit",  525, 515, 685, 555),
    HudButton("SAVE",         "S", "S. Save Lines", "Save to config", 695, 515, 835, 555),
    HudButton("RESET",        "Esc", "Esc. Cancel", "Cancel/Reset", 845, 515, 985, 555),
    HudButton("TOGGLE_MODE",  "R", "R. Mode",       "Toggle Mode", 995, 515, 1265, 555),
]


def _save_active_calibration_lines() -> bool:
    global hud_status_text, hud_status_time
    proc1 = get_processor(CAMERA_1_ID)
    proc2 = get_processor(CAMERA_2_ID)
    from line_calibration import save_calibrated_lines, get_current_lines
    lines_to_save = get_current_lines()
    if proc1:
        if proc1.entry_line:
            lines_to_save["ENTRY_LINE_1"] = proc1.entry_line
        if proc1.exit_line:
            lines_to_save["EXIT_LINE_1"] = proc1.exit_line
    if proc2:
        if proc2.entry_line:
            lines_to_save["ENTRY_LINE_2"] = proc2.entry_line
        if proc2.exit_line:
            lines_to_save["EXIT_LINE_2"] = proc2.exit_line

    success = save_calibrated_lines(lines_to_save)
    if success:
        hud_status_text = "SAVED: All 4 lines successfully saved to lines_config.json."
        print(f"[CALIBRATION SAVED] All 4 lines successfully persisted to disk: {lines_to_save}")
    else:
        hud_status_text = "ERROR: Failed to save lines to lines_config.json."
        print("[CALIBRATION ERROR] Failed to save lines to lines_config.json")
    hud_status_time = time.time()
    return success


def _cancel_active_calibration(line_key: str | None = None) -> None:
    global hud_status_text, hud_status_time, calib_click_points
    calib_click_points.clear()
    from line_calibration import load_calibrated_lines, DEFAULT_LINES
    saved = load_calibrated_lines()
    proc1 = get_processor(CAMERA_1_ID)
    proc2 = get_processor(CAMERA_2_ID)

    if line_key and line_key in DEFAULT_LINES:
        def_coords = DEFAULT_LINES[line_key]
        if line_key == "ENTRY_LINE_1" and proc1:
            proc1.entry_line = def_coords
        elif line_key == "EXIT_LINE_1" and proc1:
            proc1.exit_line = def_coords
        elif line_key == "ENTRY_LINE_2" and proc2:
            proc2.entry_line = def_coords
        elif line_key == "EXIT_LINE_2" and proc2:
            proc2.exit_line = def_coords
        _save_active_calibration_lines()
        hud_status_text = f"RESET: {LINE_NAME_MAP.get(line_key, line_key)} restored to default {def_coords}."
        print(f"[CALIBRATION RESET] {line_key} restored to default {def_coords}")
    else:
        if proc1:
            proc1.entry_line = saved.get("ENTRY_LINE_1", DEFAULT_LINES["ENTRY_LINE_1"])
            proc1.exit_line = saved.get("EXIT_LINE_1", DEFAULT_LINES["EXIT_LINE_1"])
        if proc2:
            proc2.entry_line = saved.get("ENTRY_LINE_2", DEFAULT_LINES["ENTRY_LINE_2"])
            proc2.exit_line = saved.get("EXIT_LINE_2", DEFAULT_LINES["EXIT_LINE_2"])
        hud_status_text = "CANCELLED: Unsaved changes discarded. Restored saved coordinates."
        print("[CALIBRATION CANCELLED] Restored saved coordinates from lines_config.json")
    hud_status_time = time.time()


def _reset_active_calib_line(line_key: str) -> None:
    _cancel_active_calibration(line_key)


def calib_mouse_callback(event, x, y, flags, param):
    global selected_calib_line, calib_click_points, mouse_cursor_pos
    global hud_status_text, hud_status_time, active_app_mode

    mouse_cursor_pos = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    print(f"\n[MOUSE CLICK] Window Position: (x={x}, y={y}) | Mode: {active_app_mode} | Active Line: {selected_calib_line}")

    # 1. Click on HUD Buttons (y >= 480)
    if y >= 480:
        for btn in HUD_BUTTONS:
            if btn.x1 <= x <= btn.x2 and btn.y1 <= y <= btn.y2:
                btn_id = btn.id
                print(f"[HUD BUTTON CLICKED] {btn_id} ({btn.desc})")
                if btn_id in ("ENTRY_LINE_1", "EXIT_LINE_1", "ENTRY_LINE_2", "EXIT_LINE_2"):
                    selected_calib_line = btn_id
                    calib_click_points.clear()
                    active_app_mode = "CALIBRATION"
                    hud_status_text = f"Selected {LINE_NAME_MAP[btn_id]}. Click 2 points on camera video."
                    hud_status_time = time.time()
                elif btn_id == "SAVE":
                    _save_active_calibration_lines()
                elif btn_id in ("CANCEL", "RESET"):
                    _reset_active_calib_line(selected_calib_line)
                elif btn_id == "TOGGLE_MODE":
                    if active_app_mode == "CALIBRATION":
                        active_app_mode = "DETECTION"
                        hud_status_text = "Detection Active: Processing vehicle line crossings."
                        print("[MODE] Switched to DETECTION MODE.")
                    else:
                        active_app_mode = "CALIBRATION"
                        hud_status_text = "Calibration Mode: Select [1..4] and click 2 points to draw."
                        print("[MODE] Switched to CALIBRATION MODE.")
                    hud_status_time = time.time()
                return

    # 2. Click in Video Preview Area (y < 480)
    # Check top panel selector tabs (y: 48..75 on preview)
    if 48 <= y <= 75:
        if 15 <= x <= 110:
            selected_calib_line = "ENTRY_LINE_1"
            calib_click_points.clear()
            active_app_mode = "CALIBRATION"
            hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCT panel."
            hud_status_time = time.time()
            print(f"[TAB CLICKED] Selected KRCT Entry Line (ENTRY_LINE_1)")
            return
        elif 120 <= x <= 215:
            selected_calib_line = "EXIT_LINE_1"
            calib_click_points.clear()
            active_app_mode = "CALIBRATION"
            hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCT panel."
            hud_status_time = time.time()
            print(f"[TAB CLICKED] Selected KRCT Exit Line (EXIT_LINE_1)")
            return
        elif 655 <= x <= 750:
            selected_calib_line = "ENTRY_LINE_2"
            calib_click_points.clear()
            active_app_mode = "CALIBRATION"
            hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCE panel."
            hud_status_time = time.time()
            print(f"[TAB CLICKED] Selected KRCE Entry Line (ENTRY_LINE_2)")
            return
        elif 760 <= x <= 855:
            selected_calib_line = "EXIT_LINE_2"
            calib_click_points.clear()
            active_app_mode = "CALIBRATION"
            hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCE panel."
            hud_status_time = time.time()
            print(f"[TAB CLICKED] Selected KRCE Exit Line (EXIT_LINE_2)")
            return

    # Clicking in video area activates CALIBRATION mode automatically
    if active_app_mode != "CALIBRATION":
        active_app_mode = "CALIBRATION"
        print("[MODE AUTO-SWITCH] Switched to CALIBRATION MODE via preview click.")

    is_left_click = (x < 640)
    is_krct_line = selected_calib_line in ("ENTRY_LINE_1", "EXIT_LINE_1")

    proc1 = get_processor(CAMERA_1_ID)
    proc2 = get_processor(CAMERA_2_ID)

    if is_krct_line and not is_left_click:
        hud_status_text = f"Notice: {LINE_NAME_MAP[selected_calib_line]} belongs to KRCT (Click LEFT panel)."
        hud_status_time = time.time()
        print(f"[CLICK REJECTED] Click at x={x} is on KRCE (Right panel), but active line is {selected_calib_line} (KRCT). Click Left panel or select 3/4.")
        return

    if not is_krct_line and is_left_click:
        hud_status_text = f"Notice: {LINE_NAME_MAP[selected_calib_line]} belongs to KRCE (Click RIGHT panel)."
        hud_status_time = time.time()
        print(f"[CLICK REJECTED] Click at x={x} is on KRCT (Left panel), but active line is {selected_calib_line} (KRCE). Click Right panel or select 1/2.")
        return

    target_proc = proc1 if is_krct_line else proc2
    if target_proc is None:
        print("[ERROR] Target camera processor is not initialized.")
        return

    # Relative coordinates inside the 640x480 panel
    local_x = x if is_krct_line else (x - 640)
    local_y = y

    orig_w = int(getattr(target_proc, "frame_width", 1920) or 1920)
    orig_h = int(getattr(target_proc, "frame_height", 1080) or 1080)
    if orig_w <= 0: orig_w = 1920
    if orig_h <= 0: orig_h = 1080

    scale_x = orig_w / 640.0
    scale_y = orig_h / 480.0

    cam_x = int(round(local_x * scale_x))
    cam_y = int(round(local_y * scale_y))

    cam_x = max(0, min(cam_x, orig_w - 1))
    cam_y = max(0, min(cam_y, orig_h - 1))

    disp_pt = (local_x, local_y)
    cam_pt = (cam_x, cam_y)
    target_cam = "KRCT" if is_krct_line else "KRCE"

    if len(calib_click_points) == 0:
        # First click: Start point
        calib_click_points.append((disp_pt, cam_pt))
        hud_status_text = f"FIRST POINT SET at {cam_pt}. Click End point for {LINE_NAME_MAP[selected_calib_line]}."
        hud_status_time = time.time()
        print(f"[POINT 1 SET] Camera: {target_cam} | Line: {selected_calib_line} | Display: {disp_pt} | Frame Coords: {cam_pt}")
    else:
        # Second click: End point
        start_disp, start_cam = calib_click_points[0]
        if start_cam == cam_pt:
            hud_status_text = "Validation Error: Endpoints cannot be identical. Click a different point."
            hud_status_time = time.time()
            print(f"[REJECTED] Endpoints identical at {cam_pt}. Please click a distinct point.")
            return

        new_cam_line = (start_cam, cam_pt)
        calib_click_points.clear()

        # Update active processor immediately for live crossing detection and rendering
        if selected_calib_line == "ENTRY_LINE_1" and proc1:
            proc1.entry_line = new_cam_line
        elif selected_calib_line == "EXIT_LINE_1" and proc1:
            proc1.exit_line = new_cam_line
        elif selected_calib_line == "ENTRY_LINE_2" and proc2:
            proc2.entry_line = new_cam_line
        elif selected_calib_line == "EXIT_LINE_2" and proc2:
            proc2.exit_line = new_cam_line

        # Auto-persist to lines_config.json
        _save_active_calibration_lines()

        hud_status_text = f"UPDATED: {LINE_NAME_MAP[selected_calib_line]} -> {new_cam_line}. Press [S] to Save."
        hud_status_time = time.time()
        print(f"[POINT 2 SET] Camera: {target_cam} | Line: {selected_calib_line} | Display: {disp_pt} | Frame Coords: {cam_pt}")
        print(f"[LINE COMPLETED] {selected_calib_line} configured as: {new_cam_line}")


def draw_large_status_banner(canvas: np.ndarray) -> None:
    """Draws a large, high-visibility status banner across the top of the combined preview."""
    h_banner = 45
    overlay = canvas.copy()
    now = time.time()
    just_saved = (now - hud_status_time < 3.0) and ("SAVED" in hud_status_text or "UPDATED" in hud_status_text)

    if just_saved:
        bg_col = (20, 90, 30)         # Vibrant green
        text_col = (0, 255, 200)       # Bright Cyan/Gold
        border_col = (0, 255, 100)
        line_name = LINE_NAME_MAP.get(selected_calib_line, selected_calib_line)
        msg = f"[ LINE SAVED ]  {line_name} Updated  |  Press [S] to Save  |  [R] Detection"
    elif active_app_mode == "CALIBRATION":
        if len(calib_click_points) == 1:
            bg_col = (80, 20, 80)       # Deep Purple/Magenta
            text_col = (0, 255, 255)     # Yellow
            border_col = (255, 0, 255)
            p1_cam = calib_click_points[0][1]
            line_name = LINE_NAME_MAP.get(selected_calib_line, selected_calib_line)
            msg = f"[ FIRST POINT SET at {p1_cam} ]  Click Point 2 on Video for {line_name}  |  [Esc] Cancel"
        else:
            bg_col = (25, 55, 95)       # Deep Navy Blue
            text_col = (0, 235, 255)     # Bright Cyan/Gold
            border_col = (0, 215, 255)
            line_name = LINE_NAME_MAP.get(selected_calib_line, selected_calib_line)
            msg = f"[ CALIBRATION ACTIVE ]  Editing: {line_name}  |  Left-Click Video to Draw  |  Keys: 1, 2, 3, 4"
    else:
        bg_col = (20, 45, 25)           # Dark Slate Green
        text_col = (0, 255, 120)         # Soft Green
        border_col = (0, 180, 80)
        msg = "[ NORMAL DETECTION ACTIVE ]  Buses Monitored  |  Press [R] to Enter Line Calibration  |  [Q] Quit"

    cv2.rectangle(overlay, (0, 0), (1280, h_banner), bg_col, -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.rectangle(canvas, (0, 0), (1280, h_banner), border_col, 2)

    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    tx = max(15, (1280 - tw) // 2)
    ty = (h_banner + th) // 2
    cv2.putText(canvas, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.60, text_col, 2, cv2.LINE_AA)


def render_lines_on_panel(panel, proc, is_krct, active_line_key, pending_clicks, mouse_pos):
    """
    Renders entry line (Yellow) and exit line (Orange) onto the 640x480 preview panel
    with clear endpoint visual markers, labels, and real-time drawing rubber-band preview.
    If a line is unconfigured, prominently warns that calibration is required.
    """
    if proc is None or panel is None:
        return

    orig_w = int(getattr(proc, "frame_width", 1920) or 1920)
    orig_h = int(getattr(proc, "frame_height", 1080) or 1080)
    if orig_w <= 0: orig_w = 1920
    if orig_h <= 0: orig_h = 1080

    scale_x = 640.0 / orig_w
    scale_y = 480.0 / orig_h

    def to_disp(cam_pt):
        if not cam_pt or len(cam_pt) != 2:
            return None
        dx = int(round(cam_pt[0] * scale_x))
        dy = int(round(cam_pt[1] * scale_y))
        return (dx, dy)

    college = str(getattr(proc, "college", "KRCT" if is_krct else "KRCE"))

    # Render on-panel clickable line selector tabs (y: 48 to 75)
    e_key = "ENTRY_LINE_1" if is_krct else "ENTRY_LINE_2"
    x_key = "EXIT_LINE_1" if is_krct else "EXIT_LINE_2"
    e_num = "1" if is_krct else "3"
    x_num = "2" if is_krct else "4"

    # Tab 1: Entry
    e_active = (active_line_key == e_key)
    e_bg = (0, 100, 100) if e_active else (35, 35, 45)
    cv2.rectangle(panel, (15, 48), (110, 75), e_bg, -1)
    cv2.rectangle(panel, (15, 48), (110, 75), (0, 255, 255) if e_active else (80, 80, 90), 2 if e_active else 1)
    cv2.putText(panel, f"[{e_num}] ENTRY", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255) if e_active else (200, 200, 200), 1)

    # Tab 2: Exit
    x_active = (active_line_key == x_key)
    x_bg = (0, 60, 120) if x_active else (35, 35, 45)
    cv2.rectangle(panel, (120, 48), (215, 75), x_bg, -1)
    cv2.rectangle(panel, (120, 48), (215, 75), (0, 140, 255) if x_active else (80, 80, 90), 2 if x_active else 1)
    cv2.putText(panel, f"[{x_num}] EXIT", (128, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255) if x_active else (200, 200, 200), 1)

    # 1. Draw Exit Line in Orange: (0, 140, 255)
    exit_key = "EXIT_LINE_1" if is_krct else "EXIT_LINE_2"
    is_exit_active = (active_line_key == exit_key)
    exit_coords = _line_to_points(getattr(proc, "exit_line", None))
    if exit_coords is not None:
        p1 = to_disp(exit_coords[0])
        p2 = to_disp(exit_coords[1])
        if p1 and p2:
            thickness = 4 if is_exit_active else 2
            # Orange line
            cv2.line(panel, p1, p2, (0, 140, 255), thickness)
            # Endpoint markers
            for idx, pt in enumerate([p1, p2]):
                cv2.circle(panel, pt, 7, (255, 255, 255), -1)
                cv2.circle(panel, pt, 5, (0, 140, 255), -1)
                cv2.putText(panel, str(idx + 1), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Label
            mid_x = min(p1[0], p2[0]) + 10
            mid_y = max(90, min(p1[1], p2[1]) - 10)
            tag = f"{college} EXIT" + (" [ACTIVE]" if is_exit_active else "")
            cv2.putText(panel, tag, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
    else:
        # Unconfigured line notice
        cv2.putText(panel, f"{college} EXIT: CALIBRATION REQUIRED", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    # 2. Draw Entry Line in Yellow: (0, 255, 255)
    entry_key = "ENTRY_LINE_1" if is_krct else "ENTRY_LINE_2"
    is_entry_active = (active_line_key == entry_key)
    entry_coords = _line_to_points(getattr(proc, "entry_line", None))
    if entry_coords is not None:
        p1 = to_disp(entry_coords[0])
        p2 = to_disp(entry_coords[1])
        if p1 and p2:
            thickness = 4 if is_entry_active else 2
            # Yellow line
            cv2.line(panel, p1, p2, (0, 255, 255), thickness)
            # Endpoint markers
            for idx, pt in enumerate([p1, p2]):
                cv2.circle(panel, pt, 7, (255, 255, 255), -1)
                cv2.circle(panel, pt, 5, (0, 255, 255), -1)
                cv2.putText(panel, str(idx + 1), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Label
            mid_x = min(p1[0], p2[0]) + 10
            mid_y = max(90, min(p1[1], p2[1]) - 10)
            tag = f"{college} ENTRY" + (" [ACTIVE]" if is_entry_active else "")
            cv2.putText(panel, tag, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        # Unconfigured line notice
        cv2.putText(panel, f"{college} ENTRY: CALIBRATION REQUIRED", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    # 3. If a line is currently being clicked on this panel (First point set)
    is_this_panel_active = (is_krct and active_line_key in ("ENTRY_LINE_1", "EXIT_LINE_1")) or \
                           (not is_krct and active_line_key in ("ENTRY_LINE_2", "EXIT_LINE_2"))

    if is_this_panel_active and len(pending_clicks) == 1:
        start_disp = pending_clicks[0][0]
        # Draw glowing Start Point circle
        cv2.circle(panel, start_disp, 11, (255, 0, 255), -1)
        cv2.circle(panel, start_disp, 8, (0, 255, 255), -1)
        cv2.circle(panel, start_disp, 4, (0, 0, 0), -1)
        cv2.putText(panel, "1: START", (start_disp[0] + 10, start_disp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Dynamic rubber-band line to mouse cursor
        if mouse_pos is not None:
            mx, my = mouse_pos
            if is_krct and mx < 640 and my < 480:
                cv2.line(panel, start_disp, (mx, my), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(panel, (mx, my), 6, (255, 255, 0), -1)
            elif not is_krct and mx >= 640 and my < 480:
                rel_mx = mx - 640
                cv2.line(panel, start_disp, (rel_mx, my), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(panel, (rel_mx, my), 6, (255, 255, 0), -1)


def render_hud_panel(proc1, proc2):
    """
    Renders an on-screen HUD panel below the camera feeds.
    Includes mode pill, active line indicator, status prompts, 7 interactive buttons,
    current coordinate readouts, and keyboard shortcut guide.
    """
    hud = np.full((140, 1280, 3), (24, 24, 28), dtype=np.uint8)

    # Top border line
    cv2.line(hud, (0, 0), (1280, 0), (60, 60, 75), 2)

    # ----------------------------------------------------
    # ROW 1: Mode Pill, Selected Line Pill, Status Prompt
    # ----------------------------------------------------
    if active_app_mode == "CALIBRATION":
        cv2.rectangle(hud, (15, 8), (215, 32), (20, 80, 130), -1)
        cv2.rectangle(hud, (15, 8), (215, 32), (0, 215, 255), 1)
        cv2.putText(hud, "MODE: CALIBRATION", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 235, 255), 2)
    else:
        cv2.rectangle(hud, (15, 8), (215, 32), (15, 85, 35), -1)
        cv2.rectangle(hud, (15, 8), (215, 32), (0, 255, 120), 1)
        cv2.putText(hud, "MODE: DETECTION ACTIVE", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 2)

    # Active Line Pill
    is_entry = "ENTRY" in selected_calib_line
    line_pill_color = (0, 255, 255) if is_entry else (0, 140, 255)
    cv2.rectangle(hud, (225, 8), (500, 32), (38, 38, 48), -1)
    cv2.rectangle(hud, (225, 8), (500, 32), line_pill_color, 1)
    active_line_name = LINE_NAME_MAP.get(selected_calib_line, selected_calib_line)
    cv2.putText(hud, f"ACTIVE: {active_line_name}", (235, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, line_pill_color, 2)

    # Instruction / Status Prompt
    cv2.putText(hud, hud_status_text, (510, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 245), 1)

    # ----------------------------------------------------
    # ROW 2: The 7 Interactive Buttons (y: 35 to 75 within HUD)
    # ----------------------------------------------------
    for btn in HUD_BUTTONS:
        bx1 = int(btn.x1)
        bx2 = int(btn.x2)
        hud_y1 = max(0, min(139, int(btn.y1 - 480)))
        hud_y2 = max(0, min(139, int(btn.y2 - 480)))

        btn_id = btn.id
        is_active_btn = (btn_id == selected_calib_line)

        # Background fill & border
        if btn_id == "TOGGLE_MODE":
            if active_app_mode == "CALIBRATION":
                bg_col = (20, 95, 45)
                border_col = (0, 255, 100)
                text_col = (255, 255, 255)
                lbl = "[R] Exit Calib"
            else:
                bg_col = (25, 80, 120)
                border_col = (0, 215, 255)
                text_col = (255, 255, 255)
                lbl = "[R] Calibrate Lines"
        elif btn_id == "SAVE":
            bg_col = (25, 65, 110)
            border_col = (0, 180, 255)
            text_col = (255, 255, 255)
            lbl = "[S] Save Lines"
        elif btn_id in ("CANCEL", "RESET"):
            bg_col = (50, 35, 35)
            border_col = (180, 90, 90)
            text_col = (230, 230, 230)
            lbl = "[Esc] Cancel"
        elif is_active_btn:
            is_entry_btn = "ENTRY" in btn_id
            bg_col = (60, 60, 25) if is_entry_btn else (50, 40, 15)
            border_col = (0, 255, 255) if is_entry_btn else (0, 140, 255)
            text_col = (255, 255, 255)
            lbl = f"[{btn.key}] {btn.desc}"
        else:
            bg_col = (35, 35, 42)
            border_col = (75, 75, 90)
            text_col = (200, 200, 200)
            lbl = f"[{btn.key}] {btn.desc}"

        cv2.rectangle(hud, (bx1, hud_y1), (bx2, hud_y2), bg_col, -1)
        cv2.rectangle(hud, (bx1, hud_y1), (bx2, hud_y2), border_col, 2 if is_active_btn else 1)

        # Center text inside button
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = int(bx1 + max(5, (bx2 - bx1 - tw) // 2))
        ty = int(hud_y1 + ((hud_y2 - hud_y1 + th) // 2))
        cv2.putText(hud, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_col, 1)

    # ----------------------------------------------------
    # ROW 3: Active Coordinates Summary & Keyboard Shortcuts Guide
    # ----------------------------------------------------
    e1_str = str(getattr(proc1, "entry_line", "N/A"))
    x1_str = str(getattr(proc1, "exit_line", "N/A"))
    e2_str = str(getattr(proc2, "entry_line", "N/A"))
    x2_str = str(getattr(proc2, "exit_line", "N/A"))

    coords_text = f"KRCT: Entry={e1_str} | Exit={x1_str}   ||   KRCE: Entry={e2_str} | Exit={x2_str}"
    cv2.putText(hud, coords_text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (175, 185, 195), 1)

    guide_text = "Controls: [R] Calibrate/Detection | [1..4] Select Line | Click 2 points | [S] Save | [Esc] Cancel | [Q] Quit"
    cv2.putText(hud, guide_text, (15, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 200, 240), 1)

    return hud


def start_camera():
    """
    Main entry point for starting the Dual-Camera system:
      1. Loads saved lines from lines_config.json
      2. Preloads OCR models
      3. Starts Background OCR Worker
      4. Initializes and starts KRCT (Camera 1) and KRCE (Camera 2) processors
      5. Starts Dual WebSocket Server
      6. Displays interactive dual-camera desktop view with mouse-based line calibration
    """
    global active_app_mode, selected_calib_line, calib_click_points
    global hud_status_text, hud_status_time

    print("=" * 65)
    print("IN/OUT X — STARTING DUAL-CAMERA ENGINE")
    print("=" * 65)
    print(f"Camera 1 (KRCT) : {CAMERA_1_ID} -> {format_camera_source(CAMERA_SOURCE_1)}")
    print(f"Camera 2 (KRCE) : {CAMERA_2_ID} -> {format_camera_source(CAMERA_SOURCE_2)}")
    print("=" * 65)

    # 1. Preload OCR
    preload_ocr()

    # 2. Start shared OCR worker
    start_ocr_worker()

    # 3. Initialize & start cameras
    initialize_cameras()
    for proc in camera_processors.values():
        proc.start()

    # 4. Start WebSocket Server in background
    ws_thread = threading.Thread(
        target=run_websocket_server,
        daemon=True,
        name="WebSocketServerThread"
    )
    ws_thread.start()

    # 5. Desktop Multi-view & Mouse Calibration Window (or Headless Server Mode)
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gui_enabled = False

    if has_display:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 620)
            cv2.setMouseCallback(WINDOW_NAME, calib_mouse_callback)
            gui_enabled = True
            print("\n[GUI MONITOR INITIALIZED]")
            print("[KEYBOARD & MOUSE CALIBRATION INSTRUCTIONS]")
            print("  1. Press 'R' on keyboard or click [R] button to toggle Line Calibration.")
            print("  2. Select the line to calibrate using keyboard keys 1, 2, 3, 4:")
            print("     [1] Camera 1 / KRCT Entry Line")
            print("     [2] Camera 1 / KRCT Exit Line")
            print("     [3] Camera 2 / KRCE Entry Line")
            print("     [4] Camera 2 / KRCE Exit Line")
            print("  3. Click Left Mouse Button on video to set START point (Point 1).")
            print("  4. Click Left Mouse Button on video to set END point (Point 2).")
            print("  5. Line updates immediately. Press 'S' to Save lines to lines_config.json.")
            print("  6. Press 'Esc' to cancel edits without saving.")
            print("  7. Press 'R' to return to normal Detection mode.")
            print("  8. Press 'Q' to quit.\n")
        except Exception as e:
            print(f"[GUI WARNING] Could not initialize OpenCV GUI window ({e}). Running in Headless Mode.")
            gui_enabled = False
    else:
        print("\n[HEADLESS MODE] No display detected (DISPLAY/WAYLAND_DISPLAY not set).")
        print("[HEADLESS MODE] Running AI Detection and WebSocket Server in background.")
        print("[HEADLESS MODE] To enable GUI monitor window, run: export DISPLAY=:0 before starting.\n")

    first_frame_shown = False

    try:
        while True:
            if not gui_enabled:
                time.sleep(1.0)
                continue

            proc1 = get_processor(CAMERA_1_ID)
            proc2 = get_processor(CAMERA_2_ID)

            frame1 = None
            if proc1:
                with proc1.frame_lock:
                    if proc1.latest_annotated_frame is not None:
                        frame1 = proc1.latest_annotated_frame.copy()
            if frame1 is None:
                frame1 = create_disconnected_placeholder(
                    COLLEGE_1, CAMERA_1_ID, proc1.status_message if proc1 else "Not Initialized"
                )

            frame2 = None
            if proc2:
                with proc2.frame_lock:
                    if proc2.latest_annotated_frame is not None:
                        frame2 = proc2.latest_annotated_frame.copy()
            if frame2 is None:
                frame2 = create_disconnected_placeholder(
                    COLLEGE_2, CAMERA_2_ID, proc2.status_message if proc2 else "Not Initialized"
                )

            # Resize to 640x480 each for clean side-by-side display
            target_h = 480
            target_w = 640
            f1_resized = cv2.resize(frame1, (target_w, target_h))
            f2_resized = cv2.resize(frame2, (target_w, target_h))

            # Render calibrated lines & visual markers on the resized preview
            render_lines_on_panel(f1_resized, proc1, is_krct=True,
                                  active_line_key=selected_calib_line,
                                  pending_clicks=calib_click_points,
                                  mouse_pos=mouse_cursor_pos)

            render_lines_on_panel(f2_resized, proc2, is_krct=False,
                                  active_line_key=selected_calib_line,
                                  pending_clicks=calib_click_points,
                                  mouse_pos=mouse_cursor_pos)

            # Combine video panels side-by-side (1280 x 480)
            combined_view = np.hstack([f1_resized, f2_resized])

            # Render HUD panel below video (1280 x 140)
            hud_panel = render_hud_panel(proc1, proc2)

            # Final canvas (1280 x 620)
            canvas = np.vstack([combined_view, hud_panel])

            # Draw large prominent status banner across the top of video preview
            draw_large_status_banner(canvas)

            cv2.imshow(WINDOW_NAME, canvas)

            # Ensure mouse callback is reliably hooked to the Win32 window after initial frame
            if not first_frame_shown:
                cv2.setMouseCallback(WINDOW_NAME, calib_mouse_callback)
                first_frame_shown = True

            key = cv2.waitKey(20) & 0xFF
            if key != 255:
                key_char = chr(key) if 32 <= key <= 126 else f"code_{key}"
                print(f"[KEY PRESS] Key: '{key_char}' ({key}) | Current Mode: {active_app_mode} | Line: {selected_calib_line}")

            if key in (ord("q"), ord("Q")):
                print("[SYSTEM] Shutdown requested via 'Q' key...")
                break
            elif key in (ord("r"), ord("R")):
                if active_app_mode == "CALIBRATION":
                    active_app_mode = "DETECTION"
                    calib_click_points.clear()
                    hud_status_text = "Detection Active: Processing college bus line crossings."
                    print("[MODE TOGGLE] Switched to NORMAL DETECTION MODE.")
                else:
                    active_app_mode = "CALIBRATION"
                    calib_click_points.clear()
                    hud_status_text = "Calibration Mode: Press [1..4] to select line, click 2 points to draw, [S] to save."
                    print("[MODE TOGGLE] Switched to LINE CALIBRATION MODE.")
                    print("  [1] Camera 1 / KRCT Entry Line")
                    print("  [2] Camera 1 / KRCT Exit Line")
                    print("  [3] Camera 2 / KRCE Entry Line")
                    print("  [4] Camera 2 / KRCE Exit Line")
                    print("  [S] Save lines to disk")
                    print("  [Esc] Cancel without saving")
                    print("  [R] Exit calibration mode")
                hud_status_time = time.time()
            elif key == ord("1"):
                active_app_mode = "CALIBRATION"
                selected_calib_line = "ENTRY_LINE_1"
                calib_click_points.clear()
                hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCT panel (Left)."
                hud_status_time = time.time()
                print(f"[LINE SELECTED] {LINE_NAME_MAP[selected_calib_line]} (ENTRY_LINE_1) ready for drawing.")
            elif key == ord("2"):
                active_app_mode = "CALIBRATION"
                selected_calib_line = "EXIT_LINE_1"
                calib_click_points.clear()
                hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCT panel (Left)."
                hud_status_time = time.time()
                print(f"[LINE SELECTED] {LINE_NAME_MAP[selected_calib_line]} (EXIT_LINE_1) ready for drawing.")
            elif key == ord("3"):
                active_app_mode = "CALIBRATION"
                selected_calib_line = "ENTRY_LINE_2"
                calib_click_points.clear()
                hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCE panel (Right)."
                hud_status_time = time.time()
                print(f"[LINE SELECTED] {LINE_NAME_MAP[selected_calib_line]} (ENTRY_LINE_2) ready for drawing.")
            elif key == ord("4"):
                active_app_mode = "CALIBRATION"
                selected_calib_line = "EXIT_LINE_2"
                calib_click_points.clear()
                hud_status_text = f"Selected {LINE_NAME_MAP[selected_calib_line]}. Click 2 points on KRCE panel (Right)."
                hud_status_time = time.time()
                print(f"[LINE SELECTED] {LINE_NAME_MAP[selected_calib_line]} (EXIT_LINE_2) ready for drawing.")
            elif key in (ord("s"), ord("S")):
                _save_active_calibration_lines()
            elif key == 27:  # Esc
                _cancel_active_calibration(selected_calib_line)
            elif key in (ord("d"), ord("D"), 13, 32):  # D, Enter, Space
                if active_app_mode == "CALIBRATION":
                    active_app_mode = "DETECTION"
                    hud_status_text = "Detection Active: Processing vehicle line crossings."
                    print("[MODE] Detection mode active.")
                else:
                    active_app_mode = "CALIBRATION"
                    hud_status_text = "Calibration Mode: Click to draw lines or press S to save."
                    print("[MODE] Calibration mode active.")
                hud_status_time = time.time()
            elif key in (ord("c"), ord("C")):
                active_app_mode = "CALIBRATION"
                hud_status_text = "Calibration Mode: Click to draw lines or press S to save."
                print("[MODE] Calibration mode active.")
                hud_status_time = time.time()

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user (Ctrl+C).")
    finally:
        # Graceful cleanup
        for proc in camera_processors.values():
            proc.stop()
        stop_ocr_worker()
        if gui_enabled:
            cv2.destroyAllWindows()
        print("[SYSTEM] Dual-camera engine stopped cleanly.")


if __name__ == "__main__":
    start_camera()