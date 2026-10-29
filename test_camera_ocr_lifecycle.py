import importlib.util
import queue
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


def _database_stub(*_args, **_kwargs):
    return None


_mongodb_stub = ModuleType("mongodb")
_mongodb_stub.CollegeBusVerificationError = type(
    "CollegeBusVerificationError",
    (RuntimeError,),
    {},
)
for _name in (
    "insert_bus_log",
    "find_college_bus",
    "verify_college_bus",
    "find_best_matching_bus",
    "find_bus_by_number",
    "find_staff_vehicle",
    "save_staff_vehicle",
    "save_unknown_vehicle",
):
    setattr(_mongodb_stub, _name, _database_stub)
_notification_stub = ModuleType("notification")
_notification_stub.create_notification = _database_stub

_camera_spec = importlib.util.spec_from_file_location(
    "inoutx_camera_ocr_tests",
    Path(__file__).with_name("camera.py"),
)
camera = importlib.util.module_from_spec(_camera_spec)
sys.modules[_camera_spec.name] = camera
with patch.dict(
    sys.modules,
    {"mongodb": _mongodb_stub, "notification": _notification_stub},
):
    _camera_spec.loader.exec_module(camera)

plate_detector_globals = camera.detect_plate_for_vehicle.__globals__
CameraProcessor = camera.CameraProcessor
OCRJob = camera.OCRJob
is_stable = camera.is_stable
normalize_plate_result = camera.normalize_plate_result


def make_processor(camera_id="KRCT_GATE_01", college="KRCT"):
    return CameraProcessor(
        camera_id=camera_id,
        college=college,
        gate_name=f"{college} Main Gate",
        source="0",
        entry_line=((100, 200), (300, 200)),
        exit_line=((400, 200), (600, 200)),
        enabled=False,
        tracker=object(),
    )


def stable_state(processor, track_id, plate):
    state = processor._ensure_ocr_state(track_id)
    state["plates"] = [plate, plate]
    return state


class FakeModelBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = np.array([class_id], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.xyxy = np.array([coordinates], dtype=np.float32)


class FakeModelBoxes:
    def __init__(self, boxes):
        self.items = boxes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class FakeModelResult:
    def __init__(self, boxes):
        self.boxes = FakeModelBoxes(boxes)


class FakeDetectorModel:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes
        self.inputs = []

    def predict(self, image, **_kwargs):
        self.inputs.append(image)
        return [FakeModelResult(self.boxes)]


def test_stability_uses_normalized_matching_results():
    assert normalize_plate_result(" tn-24 k.6050 ") == "TN24K6050"
    assert is_stable(["tn-24 k.6050", "TN24K6050"])
    assert not is_stable(["TN24K6050", "TN24K605O"])
    assert not is_stable(["TN24K6050", "TN24K6050", "TN24K605O", "TN24K605O"])


def test_best_model_detects_named_rear_and_plate_classes(monkeypatch):
    model = FakeDetectorModel(
        {0: "bus_rear", 1: "license_plate"},
        [
            FakeModelBox(0, 0.96, (5, 5, 95, 95)),
            FakeModelBox(1, 0.87, (25, 65, 75, 82)),
            FakeModelBox(1, 0.10, (30, 65, 70, 80)),
        ],
    )
    monkeypatch.setitem(plate_detector_globals, "bus_plate_model", model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = plate_detector_globals["detect_bus_rear_and_plates"](image)

    assert len(model.inputs) == 1
    assert detections["bus_rear"] == [
        {"box": (5, 5, 95, 95), "confidence": np.float32(0.96)}
    ]
    assert len(detections["license_plate"]) == 1
    assert detections["license_plate"][0]["box"] == (25, 65, 75, 82)
    assert detections["license_plate"][0]["confidence"] >= 0.20


def test_only_license_plate_crop_reaches_existing_crop_selection(monkeypatch):
    model = FakeDetectorModel(
        {0: "bus_rear", 1: "license_plate"},
        [
            FakeModelBox(0, 0.96, (2, 2, 98, 98)),
            FakeModelBox(1, 0.90, (30, 60, 75, 78)),
        ],
    )
    monkeypatch.setitem(plate_detector_globals, "bus_plate_model", model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[60:78, 30:75] = 255

    plate_results = camera.detect_plate_for_vehicle(image)
    crop, plate_box, confidence = camera.get_best_plate_crop(image, plate_results)

    assert len(plate_results) == 1
    assert plate_box == (30, 60, 75, 78)
    assert confidence >= 0.20
    assert crop is not None
    assert crop.shape[0] < image.shape[0]
    assert crop.shape[1] < image.shape[1]
    assert np.max(crop) == 255


def test_bus_rear_detection_limits_legacy_fallback_to_rear_crop(monkeypatch):
    model = FakeDetectorModel(
        {0: "bus_rear", 1: "license_plate"},
        [FakeModelBox(0, 0.92, (20, 20, 60, 60))],
    )
    monkeypatch.setitem(plate_detector_globals, "bus_plate_model", model)
    legacy_input_shapes = []

    def legacy_detection(image):
        legacy_input_shapes.append(image.shape[:2])
        return [{"box": (2, 2, 20, 10), "confidence": 0.75}]

    monkeypatch.setitem(plate_detector_globals, "detect_plate", legacy_detection)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = camera.detect_plate_for_vehicle(image)

    assert legacy_input_shapes == [(44, 44)]
    assert detections == [{"box": (20, 20, 38, 28), "confidence": 0.75}]


def test_legacy_detector_remains_fallback_without_best_model(monkeypatch):
    monkeypatch.setitem(plate_detector_globals, "bus_plate_model", None)
    legacy_calls = []
    expected = [{"box": (10, 10, 50, 25), "confidence": 0.8}]
    monkeypatch.setitem(
        plate_detector_globals,
        "detect_plate",
        lambda image: legacy_calls.append(image) or expected,
    )
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    assert camera.detect_plate_for_vehicle(image) == expected
    assert legacy_calls == [image]


def test_database_verification_waits_for_two_matching_results(monkeypatch):
    processor = make_processor()
    state = processor._ensure_ocr_state(10)
    data = {"vehicle_type": "BUS", "movement_status": "ENTRY"}
    lookups = []

    def verify(plate, plausible_candidates=None):
        lookups.append((plate, plausible_candidates))
        return None

    monkeypatch.setattr(camera, "verify_college_bus", verify)
    state["plates"] = ["TN-24K6050"]
    assert not processor.classify_and_save(np.zeros((8, 8, 3)), 10, data, 1.0)
    assert lookups == []

    state["plates"].append("TN24K605O")
    assert not processor.classify_and_save(np.zeros((8, 8, 3)), 10, data, 1.1)
    assert lookups == []

    state["plates"].append("TN 24 K 6050")
    assert not processor.classify_and_save(np.zeros((8, 8, 3)), 10, data, 1.2)
    assert lookups == [("TN24K6050", ["TN24K6050", "TN24K6050"])]
    assert not state["terminal"]
    assert "TN24K6050" in state["rejected_plates"]
    assert state["outcome"] is None


def test_database_verification_failures_use_bounded_retries(monkeypatch):
    processor = make_processor()
    state = stable_state(processor, 13, "TN24K6050")
    verification_calls = []
    monkeypatch.setattr(camera, "DB_VERIFICATION_RETRY_INTERVAL", 0.0)

    def failed_verification(*args, **kwargs):
        verification_calls.append(1)
        raise camera.CollegeBusVerificationError("offline test failure")

    monkeypatch.setattr(camera, "verify_college_bus", failed_verification)
    frame = np.zeros((8, 8, 3))
    data = {"vehicle_type": "BUS", "movement_status": "ENTRY"}

    for attempt in range(camera.DB_VERIFICATION_RETRIES + 1):
        assert not processor.classify_and_save(frame, 13, data, float(attempt))

    assert len(verification_calls) == camera.DB_VERIFICATION_RETRIES
    assert "TN24K6050" in state["rejected_plates"]
    assert not state["terminal"]


def test_ocr_result_callback_requires_matching_reads_before_verification(monkeypatch):
    processor = make_processor()
    state = processor._ensure_ocr_state(11)
    calls = []
    monkeypatch.setattr(
        camera,
        "verify_college_bus",
        lambda plate, plausible_candidates=None: calls.append(plate) or None,
    )
    data = {"vehicle_type": "BUS", "movement_status": "ENTRY"}
    frame = np.zeros((8, 8, 3))

    processor.on_ocr_result(11, "TN24K6050", None, 0.9, "BUS")
    assert not processor.classify_and_save(frame, 11, data, 1.0)
    assert calls == []

    processor.on_ocr_result(11, "tn-24 k.6050", None, 0.9, "BUS")
    assert not processor.classify_and_save(frame, 11, data, 1.1)
    assert calls == ["TN24K6050"]
    assert "TN24K6050" in state["rejected_plates"]


def test_inconsistent_ocr_can_recover_with_two_later_matching_results(monkeypatch):
    processor = make_processor()
    state = processor._ensure_ocr_state(12)
    calls = []
    monkeypatch.setattr(
        camera,
        "verify_college_bus",
        lambda plate, plausible_candidates=None: calls.append(plate) or None,
    )
    frame = np.zeros((8, 8, 3))
    data = {"vehicle_type": "BUS", "movement_status": "ENTRY"}

    for plate in ("TN24K6050", "TN24K605O", "TN47M6477", "TN 47 M 6477"):
        processor.on_ocr_result(12, plate, None, 0.9, "BUS")
        processor.classify_and_save(frame, 12, data, 1.0)

    assert calls == ["TN47M6477"]
    assert "TN47M6477" in state["rejected_plates"]


def test_queue_full_does_not_consume_attempt_and_preserves_retry(monkeypatch):
    processor = make_processor()
    full_queue = queue.Queue(maxsize=1)
    full_queue.put(object())
    monkeypatch.setattr(camera, "ocr_queue", full_queue)
    monkeypatch.setattr(
        camera.ocr_queue,
        "put_nowait",
        lambda item: (_ for _ in ()).throw(queue.Full()),
    )
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    candidate = (frame, (10, 10, 90, 90))

    assert not processor.submit_ocr_job(
        20,
        vehicle_type="BUS",
        candidates=[candidate],
    )

    state = processor.ocr_state[20]
    assert state["attempts"] == 0
    assert state["submitted"] == 0
    assert not state["pending"]
    assert not processor.finalize_if_ocr_exhausted(20)
    assert not state["terminal"]

    available_queue = queue.Queue(maxsize=1)
    monkeypatch.setattr(camera, "ocr_queue", available_queue)
    assert processor.submit_ocr_job(
        20,
        vehicle_type="BUS",
        candidates=[candidate],
    )
    assert state["attempts"] == 1
    assert state["pending"]

def test_unlimited_mode_allows_more_than_one_hundred_attempts(monkeypatch):
    processor = make_processor()
    monkeypatch.setattr(camera, "MAX_OCR_FRAMES", 0)
    work_queue = queue.Queue()
    monkeypatch.setattr(camera, "ocr_queue", work_queue)
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    candidate = (frame, (10, 10, 90, 90))
    state = processor._ensure_ocr_state(21)

    for _ in range(101):
        assert processor.submit_ocr_job(
            21,
            vehicle_type="BUS",
            candidates=[candidate],
        )
        work_queue.get_nowait()
        processor.on_ocr_result(
            21,
            "",
            None,
            0.0,
            "BUS",
            generation=state["generation"],
        )

    assert state["attempts"] == 101
    assert state["completed"] == 101
    assert not state["pending"]
    assert not state["terminal"]


def test_candidate_collection_restarts_after_each_ocr_result(monkeypatch):
    processor = make_processor()
    processor.entry_line = ((0, 50), (100, 50))
    processor.exit_line = ((0, 0), (100, 0))
    processor.running = True
    fake_clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        camera,
        "time",
        SimpleNamespace(
            time=lambda: fake_clock.now,
            monotonic=lambda: fake_clock.now,
            sleep=lambda _delay: None,
        ),
    )
    monkeypatch.setattr(camera, "OCR_INTERVAL", 0.2)
    monkeypatch.setattr(camera, "BEST_FRAME_COLLECTION_SECONDS", 0.65)
    frames = [
        np.full((100, 100, 3), index, dtype=np.uint8)
        for index in range(12)
    ]
    next_frame = iter(frames)

    def get_next_frame():
        try:
            return next(next_frame)
        except StopIteration:
            processor.running = False
            return None

    processor.get_latest_frame = get_next_frame
    submitted_batches = []

    def submit(track_id, frame=None, vehicle_box=None, vehicle_type="BUS",
               priority=1, candidates=None):
        state = processor.ocr_state[track_id]
        state["attempts"] += 1
        state["submitted"] += 1
        state["pending"] = True
        state["last_submit"] = fake_clock.now
        submitted_batches.append(
            [int(candidate[0][0, 0, 0]) for candidate in candidates]
        )
        return True

    processor.submit_ocr_job = submit

    class FakeBox:
        id = np.array([1])
        cls = np.array([5])
        xyxy = np.array([[10, 10, 90, 90]])

    class FakeBoxes:
        id = np.array([1])

        def __iter__(self):
            return iter([FakeBox()])

    class FakeResult:
        boxes = FakeBoxes()

        def __init__(self, frame):
            self.frame = frame

        def plot(self):
            return self.frame.copy()

    class FakeTracker:
        calls = 0

        def track(self, frame):
            self.calls += 1
            if self.calls == 5:
                processor.on_ocr_result(1, "TN24K6050", None, 0.9, "BUS")
            fake_clock.now += 0.25
            return [FakeResult(frame)]

    processor.tracker = FakeTracker()
    processor._process_worker()

    assert submitted_batches == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_fast_line_crossing_uses_previous_to_current_track_segment():
    processor = make_processor()
    processor.entry_line = ((0, 50), (100, 50))
    processor.exit_line = ((0, 0), (100, 0))

    assert processor.check_first_touched_line(
        (40, 10, 60, 30),
        (40, 70, 60, 90),
    ) == "ENTRY"


def test_fair_ocr_queue_schedules_both_camera_streams():
    work_queue = camera.FairOCRQueue(
        ("KRCT_GATE_01", "KRCE_GATE_01"),
        maxsize=3,
    )
    jobs = []
    for camera_id in ("KRCT_GATE_01", "KRCT_GATE_01", "KRCE_GATE_01"):
        processor = make_processor(camera_id, "KRCE" if "KRCE" in camera_id else "KRCT")
        job = OCRJob(
            priority=1,
            job_id=len(jobs),
            payload={"camera_proc": processor, "track_id": len(jobs)},
        )
        work_queue.put_nowait(job)
        jobs.append(job)

    assert [work_queue.get(block=False).payload["camera_proc"].camera_id for _ in jobs] == [
        "KRCT_GATE_01",
        "KRCE_GATE_01",
        "KRCT_GATE_01",
    ]


def test_pre_crossing_frames_are_included_in_first_ocr_batch(monkeypatch):
    processor = make_processor()
    processor.entry_line = ((0, 50), (100, 50))
    processor.exit_line = ((0, 0), (100, 0))
    processor.running = True
    fake_clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        camera,
        "time",
        SimpleNamespace(
            time=lambda: fake_clock.now,
            monotonic=lambda: fake_clock.now,
            sleep=lambda _delay: None,
        ),
    )
    frames = [
        np.full((100, 100, 3), index, dtype=np.uint8)
        for index in range(5)
    ]
    boxes = [
        (10, 10, 90, 30),
        (10, 25, 90, 45),
        (10, 40, 90, 60),
        (10, 50, 90, 70),
        (10, 60, 90, 80),
    ]
    submitted_batches = []

    def get_next_frame():
        if not frames:
            return None
        return frames.pop(0)

    processor.get_latest_frame = get_next_frame

    def submit(track_id, frame=None, vehicle_box=None, vehicle_type="BUS",
               priority=1, candidates=None):
        state = processor.ocr_state[track_id]
        state["attempts"] += 1
        state["submitted"] += 1
        state["pending"] = True
        state["last_submit"] = fake_clock.now
        submitted_batches.append(
            [int(candidate[0][0, 0, 0]) for candidate in candidates]
        )
        return True

    processor.submit_ocr_job = submit

    class FakeBox:
        id = np.array([3])
        cls = np.array([5])
        xyxy = np.array([[0, 0, 0, 0]])

        def __init__(self, coordinates):
            self.xyxy = np.array([coordinates])

    class FakeBoxes:
        id = np.array([3])

        def __init__(self, coordinates):
            self.box = FakeBox(coordinates)

        def __iter__(self):
            return iter([self.box])

    class FakeResult:
        def __init__(self, frame, coordinates):
            self.boxes = FakeBoxes(coordinates)
            self.frame = frame

        def plot(self):
            return self.frame.copy()

    class FakeTracker:
        calls = 0

        def track(self, frame):
            coordinates = boxes[self.calls]
            self.calls += 1
            fake_clock.now += 0.25
            if self.calls == 5:
                processor.running = False
            return [FakeResult(frame, coordinates)]

    processor.tracker = FakeTracker()
    processor._process_worker()

    assert submitted_batches == [[0, 1, 2, 3]]


def test_queue_saturation_refreshes_candidates_without_dropping_crossing_frame(monkeypatch):
    processor = make_processor()
    processor.running = True
    fake_clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        camera,
        "time",
        SimpleNamespace(
            time=lambda: fake_clock.now,
            monotonic=lambda: fake_clock.now,
            sleep=lambda _delay: None,
        ),
    )
    monkeypatch.setattr(camera, "OCR_INTERVAL", 0.1)
    frame_zero = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (0, 0, 100, 100)
    state = processor._ensure_ocr_state(4)
    processor.vehicle_data[4] = {
        "last_seen": 100.0,
        "vehicle_type": "BUS",
        "previous_box": box,
        "current_box": box,
        "first_crossed_line": "ENTRY",
        "movement_status": "ENTRY",
        "crossing_time": datetime.now(),
        "identity_enabled": True,
        "processed": False,
        "candidates": [(100.0, frame_zero, box)],
        "collection_start": 100.0,
        "last_candidate_time": 0.0,
        "frame_buffer": deque(
            [(100.0, frame_zero, box)],
            maxlen=camera.PRE_CROSSING_FRAME_COUNT,
        ),
        "last_frame": frame_zero,
    }

    class FullQueue:
        def put_nowait(self, _job):
            raise queue.Full

    monkeypatch.setattr(camera, "ocr_queue", FullQueue())
    frames = [
        np.full((100, 100, 3), value, dtype=np.uint8)
        for value in range(1, 7)
    ]

    def get_next_frame():
        if not frames:
            return None
        return frames.pop(0)

    processor.get_latest_frame = get_next_frame

    class FakeBox:
        id = np.array([4])
        cls = np.array([5])
        xyxy = np.array([[0, 0, 100, 100]])

    class FakeBoxes:
        id = np.array([4])

        def __iter__(self):
            return iter([FakeBox()])

    class FakeResult:
        boxes = FakeBoxes()

        def __init__(self, frame):
            self.frame = frame

        def plot(self):
            return self.frame.copy()

    class FakeTracker:
        calls = 0

        def track(self, frame):
            self.calls += 1
            fake_clock.now += 0.25
            if self.calls == 6:
                processor.running = False
            return [FakeResult(frame)]

    processor.tracker = FakeTracker()
    processor._process_worker()

    candidates = processor.vehicle_data[4]["candidates"]
    assert len(candidates) == camera.BEST_FRAME_CANDIDATES
    assert int(candidates[0][1][0, 0, 0]) == 0
    assert [int(item[1][0, 0, 0]) for item in candidates[-3:]] == [4, 5, 6]
    assert state["attempts"] == 0
    assert not state["pending"]


def test_unlimited_retry_state_is_removed_when_track_expires(monkeypatch):
    processor = make_processor()
    processor.running = True
    fake_clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        camera,
        "time",
        SimpleNamespace(
            time=lambda: fake_clock.now,
            monotonic=lambda: fake_clock.now,
            sleep=lambda _delay: None,
        ),
    )
    state = processor._ensure_ocr_state(92)
    state["attempts"] = 101
    processor.vehicle_data[92] = {"last_seen": 99.0}
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]

    def get_next_frame():
        if not frames:
            return None
        return frames.pop(0)

    processor.get_latest_frame = get_next_frame

    class FakeTracker:
        calls = 0

        def track(self, _frame):
            self.calls += 1
            fake_clock.now += 0.6
            if self.calls == 4:
                processor.running = False
            return []

    processor.tracker = FakeTracker()
    processor._process_worker()

    assert 92 not in processor.vehicle_data
    assert 92 not in processor.ocr_state


def run_worker_job(monkeypatch, processor, track_id, candidates):
    work_queue = queue.PriorityQueue()
    monkeypatch.setattr(camera, "ocr_queue", work_queue)
    state = processor._ensure_ocr_state(track_id)
    state["pending"] = True
    state["attempts"] = 1
    state["submitted"] = 1
    job = OCRJob(
        priority=1,
        job_id=1,
        payload={
            "camera_proc": processor,
            "track_id": track_id,
            "candidates": candidates,
            "vehicle_type": "BUS",
        },
    )
    work_queue.put_nowait(job)
    completed = threading.Event()
    original_result = processor.on_ocr_result

    def on_result(**kwargs):
        original_result(**kwargs)
        completed.set()
        camera.ocr_worker_running = False

    monkeypatch.setattr(processor, "on_ocr_result", on_result)
    monkeypatch.setattr(camera, "ocr_worker_running", True)
    worker = threading.Thread(target=camera._ocr_worker_loop)
    worker.start()
    assert completed.wait(3.0)
    worker.join(timeout=3.0)
    assert not worker.is_alive()
    return state


def test_worker_exception_completes_pending_attempt(monkeypatch):
    processor = make_processor()
    frame = np.ones((64, 64, 3), dtype=np.uint8)

    def broken_detection(_crop):
        raise RuntimeError("detector failure")

    monkeypatch.setattr(camera, "detect_plate_for_vehicle", broken_detection)
    state = run_worker_job(monkeypatch, processor, 30, [(frame, (0, 0, 64, 64))])

    assert not state["pending"]
    assert state["completed"] == 1
    assert state["plates"] == []


def test_worker_selects_the_sharpest_detected_plate_candidate(monkeypatch):
    processor = make_processor()
    flat = np.zeros((64, 64, 3), dtype=np.uint8)
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    sharp = np.repeat(checker[:, :, None], 3, axis=2)
    detections = []
    selected_plate_crop = []

    def detect(_crop):
        detections.append(1)
        return [{"box": (0, 0, 128, 128), "confidence": 0.9}]

    def recognize(crop):
        selected_plate_crop.append(crop.copy())
        return "TN24K6050"

    monkeypatch.setattr(camera, "detect_plate_for_vehicle", detect)
    monkeypatch.setattr(camera, "read_plate", recognize)
    state = run_worker_job(
        monkeypatch,
        processor,
        31,
        [(flat, (0, 0, 64, 64)), (sharp, (0, 0, 64, 64))],
    )

    assert len(detections) == 2
    assert len(selected_plate_crop) == 1
    selected_sharpness = cv2.Laplacian(
        cv2.cvtColor(selected_plate_crop[0], cv2.COLOR_BGR2GRAY),
        cv2.CV_64F,
    ).var()
    assert selected_sharpness > 0
    assert state["plates"] == ["TN24K6050"]
    assert not state["pending"]


def test_duplicate_verification_and_logging_are_one_shot(monkeypatch):
    processor = make_processor()
    state = stable_state(processor, 40, "TN24K6050")
    verify_calls = []
    log_calls = []

    monkeypatch.setattr(
        camera,
        "verify_college_bus",
        lambda plate, plausible_candidates=None: (
            verify_calls.append(plate)
            or {"bus_no": "1", "plate": "TN24K6050", "status": "ACTIVE"}
        ),
    )
    monkeypatch.setattr(
        camera,
        "insert_bus_log",
        lambda data: log_calls.append(data) or "log-id",
    )
    monkeypatch.setattr(camera, "save_detection_image", lambda frame, track_id: "")
    monkeypatch.setattr(camera, "create_notification", lambda **kwargs: None)
    frame = np.zeros((8, 8, 3))
    data = {
        "vehicle_type": "BUS",
        "movement_status": "ENTRY",
        "crossing_time": datetime.now(),
    }

    assert processor.classify_and_save(frame, 40, data, 1.0)
    assert processor.classify_and_save(frame, 40, data, 2.0)
    assert state["terminal"]
    assert verify_calls == ["TN24K6050"]
    assert len(log_calls) == 1


def test_unregistered_bus_is_never_logged(monkeypatch):
    processor = make_processor()
    stable_state(processor, 41, "TN24K6050")
    log_calls = []
    monkeypatch.setattr(camera, "verify_college_bus", lambda *args, **kwargs: None)
    monkeypatch.setattr(camera, "insert_bus_log", lambda data: log_calls.append(data))

    assert not processor.classify_and_save(
        np.zeros((8, 8, 3)),
        41,
        {"vehicle_type": "BUS", "movement_status": "ENTRY"},
        1.0,
    )
    assert log_calls == []
    assert processor.ocr_state[41]["outcome"] is None


def test_bus_log_insert_failure_is_not_reported_as_success(monkeypatch):
    processor = make_processor()
    state = stable_state(processor, 42, "TN24K6050")
    monkeypatch.setattr(
        camera,
        "verify_college_bus",
        lambda *args, **kwargs: {
            "bus_no": "1",
            "plate": "TN24K6050",
            "status": "ACTIVE",
        },
    )
    monkeypatch.setattr(
        camera,
        "insert_bus_log",
        lambda data: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(camera, "save_detection_image", lambda frame, track_id: "")

    assert not processor.classify_and_save(
        np.zeros((8, 8, 3)),
        42,
        {"vehicle_type": "BUS", "movement_status": "ENTRY"},
        1.0,
    )
    assert state["terminal"]
    assert state["outcome"] == "log_failed"


def test_krct_and_krce_events_verify_concurrently(monkeypatch):
    processors = [
        (make_processor("KRCT_GATE_01", "KRCT"), "TN24K6050", 50),
        (make_processor("KRCE_GATE_01", "KRCE"), "TN47M6477", 51),
    ]
    for processor, plate, track_id in processors:
        stable_state(processor, track_id, plate)

    both_verifying = threading.Barrier(2)
    log_calls = []
    errors = []

    def verify(plate, plausible_candidates=None):
        both_verifying.wait(timeout=3.0)
        return {"bus_no": plate[-4:], "plate": plate, "status": "ACTIVE"}

    monkeypatch.setattr(camera, "verify_college_bus", verify)
    monkeypatch.setattr(
        camera,
        "insert_bus_log",
        lambda data: log_calls.append(data) or "log-id",
    )
    monkeypatch.setattr(camera, "save_detection_image", lambda frame, track_id: "")
    monkeypatch.setattr(camera, "create_notification", lambda **kwargs: None)

    def process(processor, plate, track_id):
        try:
            assert processor.classify_and_save(
                np.zeros((8, 8, 3)),
                track_id,
                {"vehicle_type": "BUS", "movement_status": "ENTRY"},
                1.0,
            )
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=process, args=(processor, plate, track_id))
        for processor, plate, track_id in processors
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert {data["camera_id"] for data in log_calls} == {
        "KRCT_GATE_01",
        "KRCE_GATE_01",
    }
