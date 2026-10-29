# ============================================================
# INOUTX - LICENSE PLATE DETECTOR
# ============================================================

from pathlib import Path

from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "license_plate_detector.pt"
)

BUS_PLATE_MODEL_PATH = BASE_DIR / "models" / "best.pt"


# ============================================================
# CONFIGURATION
# ============================================================

PLATE_CONFIDENCE = 0.20
PLATE_IOU = 0.45
PLATE_IMAGE_SIZE = 640
MAX_DETECTIONS = 5
BUS_REAR_CONFIDENCE = 0.20

MIN_ASPECT_RATIO = 1.20
MAX_ASPECT_RATIO = 6.00


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("IN/OUT X - Loading license plate model")
print("Model:", MODEL_PATH)
print("=" * 60)

try:

    plate_model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "License plate model loaded successfully."
    )

except Exception as e:

    print("=" * 60)
    print("LICENSE PLATE MODEL LOAD ERROR")
    print(e)
    print("=" * 60)

    plate_model = None


print("=" * 60)


print("=" * 60)
print("INOUTX - Loading bus-rear and license-plate model")
print("Model:", BUS_PLATE_MODEL_PATH)
print("=" * 60)

try:
    bus_plate_model = YOLO(str(BUS_PLATE_MODEL_PATH))
    print("Bus-rear and license-plate model loaded successfully.")
except Exception as e:
    print("Bus-rear and license-plate model load error:", e)
    bus_plate_model = None


def _model_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id)
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return names[class_id]
    return None


def _valid_coordinates(box, image_width, image_height):
    try:
        x1, y1, x2, y2 = map(int, box)
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def detect_bus_rear_and_plates(image):
    """Return detections grouped by the model's named object classes."""
    detections = {"bus_rear": [], "license_plate": []}
    if image is None or getattr(image, "size", 0) == 0 or bus_plate_model is None:
        return detections

    try:
        results = bus_plate_model.predict(
            image,
            conf=min(BUS_REAR_CONFIDENCE, PLATE_CONFIDENCE),
            iou=PLATE_IOU,
            imgsz=PLATE_IMAGE_SIZE,
            max_det=MAX_DETECTIONS * 2,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return detections

        image_height, image_width = image.shape[:2]
        result = results[0]
        for index in range(len(result.boxes)):
            try:
                box = result.boxes[index]
                class_id = int(box.cls[0].item())
                class_name = _model_class_name(bus_plate_model.names, class_id)
                if class_name not in detections:
                    continue
                confidence = float(box.conf[0].item())
                minimum_confidence = (
                    BUS_REAR_CONFIDENCE
                    if class_name == "bus_rear"
                    else PLATE_CONFIDENCE
                )
                if confidence < minimum_confidence:
                    continue
                coordinates = _valid_coordinates(
                    box.xyxy[0].tolist(),
                    image_width,
                    image_height,
                )
                if coordinates is None:
                    continue
                if class_name == "license_plate" and not is_valid_plate_box(
                    *coordinates,
                    confidence,
                ):
                    continue
                detections[class_name].append(
                    {"box": coordinates, "confidence": confidence}
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
    except Exception as e:
        print("Bus-rear and license-plate detection error:", e)

    detections["bus_rear"].sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )
    detections["license_plate"].sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )
    return detections


def detect_plate_for_vehicle(vehicle_crop):
    """Use best.pt for plate crops, retaining the legacy detector as fallback."""
    if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
        return []

    detections = detect_bus_rear_and_plates(vehicle_crop)
    if detections["license_plate"]:
        return detections["license_plate"]

    height, width = vehicle_crop.shape[:2]
    fallback_crop = vehicle_crop
    offset_x = 0
    offset_y = 0
    if detections["bus_rear"]:
        x1, y1, x2, y2 = detections["bus_rear"][0]["box"]
        pad_x = max(2, int((x2 - x1) * 0.05))
        pad_y = max(2, int((y2 - y1) * 0.05))
        offset_x = max(0, x1 - pad_x)
        offset_y = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)
        fallback_crop = vehicle_crop[offset_y:crop_y2, offset_x:crop_x2]
        if fallback_crop.size == 0:
            fallback_crop = vehicle_crop
            offset_x = 0
            offset_y = 0

    legacy_detections = detect_plate(fallback_crop)
    shifted_detections = []
    for detection in legacy_detections:
        coordinates = detection.get("box")
        if coordinates is None:
            continue
        x1, y1, x2, y2 = coordinates
        shifted_detections.append(
            {
                **detection,
                "box": (
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ),
            }
        )
    return shifted_detections


# ============================================================
# VALIDATE PLATE BOX
# ============================================================

def is_valid_plate_box(
    x1,
    y1,
    x2,
    y2,
    confidence
):

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    aspect_ratio = width / height

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    if confidence < PLATE_CONFIDENCE:
        return False

    return True


# ============================================================
# DETECT PLATE INSIDE VEHICLE CROP
# ============================================================

def detect_plate(vehicle_crop):
    """
    Detect number plate inside a vehicle crop.

    Input:
        vehicle_crop

    Output:
        List of dictionaries:

        [
            {
                "box": (x1, y1, x2, y2),
                "confidence": 0.91
            }
        ]

    Coordinates are relative to vehicle_crop.
    """

    if vehicle_crop is None:
        return []

    if getattr(vehicle_crop, "size", 0) == 0:
        return []

    if plate_model is None:
        return []

    try:

        results = plate_model.predict(
            vehicle_crop,
            conf=PLATE_CONFIDENCE,
            iou=PLATE_IOU,
            imgsz=PLATE_IMAGE_SIZE,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        if len(result.boxes) == 0:
            return []

        candidates = []

        for index in range(len(result.boxes)):

            try:

                box = result.boxes[index]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(
                    box.conf[0].item()
                )

                if not is_valid_plate_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                ):
                    continue

                candidates.append(
                    {
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        "confidence": confidence,
                    }
                )

            except Exception:
                continue

        candidates.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        return candidates

    except Exception as e:

        print(
            "Plate detection error:",
            e
        )

        return []


# ============================================================
# CONVERT VEHICLE-CROP BOX TO FULL-FRAME BOX
# ============================================================

def plate_box_to_frame(
    plate_box,
    vehicle_box
):
    """
    Convert plate coordinates from vehicle crop
    into original camera-frame coordinates.

    plate_box:
        (px1, py1, px2, py2)

    vehicle_box:
        (vx1, vy1, vx2, vy2)
    """

    if not plate_box:
        return None

    if not vehicle_box:
        return None

    try:

        px1, py1, px2, py2 = plate_box

        vx1, vy1, vx2, vy2 = vehicle_box

        return (
            vx1 + px1,
            vy1 + py1,
            vx1 + px2,
            vy1 + py2,
        )

    except Exception:

        return None


# ============================================================
# GET BEST PLATE
# ============================================================

def get_best_plate_detection(
    detections
):

    if not detections:
        return None

    return max(
        detections,
        key=lambda item: item.get(
            "confidence",
            0.0
        )
    )