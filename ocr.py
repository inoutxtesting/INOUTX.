# ============================================================
# INOUTX - FAST OCR MODULE
# ============================================================

import cv2
import re
import threading

from paddleocr import PaddleOCR


# ============================================================
# GLOBAL OCR MODEL
# ============================================================

ocr = None
ocr_lock = threading.Lock()


# ============================================================
# OCR CONFIGURATION
# ============================================================

FAST_ACCEPT_SCORE = 0.55
FALLBACK_ACCEPT_SCORE = 0.50

# Small plate crops need enlargement.
PLATE_SCALE = 6

# Bus body number does not need aggressive enlargement.
BUS_SCALE = 2


# ============================================================
# LOAD OCR
# ============================================================

def get_ocr():

    global ocr

    if ocr is None:

        print("=" * 60)
        print("Loading PaddleOCR...")
        print("=" * 60)

        try:

            ocr = PaddleOCR(
                lang="en",

                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,

                enable_mkldnn=False
            )

            print("PaddleOCR loaded successfully.")

        except Exception as e:

            print(
                "PaddleOCR loading failed:",
                e
            )

            ocr = None

    return ocr


# ============================================================
# PRELOAD OCR
# ============================================================

def preload_ocr():

    """
    Load PaddleOCR during application startup.

    This prevents the first detected vehicle from
    waiting for PaddleOCR model initialization.
    """

    print("=" * 60)
    print("INOUTX - PRELOADING OCR")
    print("=" * 60)

    model = get_ocr()

    if model is not None:

        print("OCR is READY before vehicle detection.")

    else:

        print("WARNING: OCR is NOT READY.")

    print("=" * 60)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_plate_text(text):

    if text is None:
        return ""

    text = str(text).upper()

    text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace("-", "")
    text = text.replace("_", "")

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# VALIDATE PLATE
# ============================================================

def is_valid_plate(text):

    if not text:
        return False

    text = clean_plate_text(text)

    # Indian registration plates normally contain
    # both letters and numbers.
    if not (7 <= len(text) <= 12):
        return False

    has_letter = any(
        char.isalpha()
        for char in text
    )

    has_number = any(
        char.isdigit()
        for char in text
    )

    return (
        has_letter
        and
        has_number
    )


# ============================================================
# RUN RAW OCR
# ============================================================

def run_ocr(image):

    if image is None:
        return []

    if image.size == 0:
        return []

    try:

        ocr_model = get_ocr()

        if ocr_model is None:
            return []

        # PaddleOCR expects a 3-channel image.
        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        # Serialize calls to the shared PaddleOCR model to avoid concurrent
        # CPU-heavy inference from multiple code paths.
        with ocr_lock:
            results = ocr_model.predict(image)

        if not results:
            return []

        result = results[0]

        # ----------------------------------------------------
        # PaddleOCR 3.x result
        # ----------------------------------------------------

        try:

            texts = result.get(
                "rec_texts",
                []
            )

        except Exception:

            texts = []

        try:

            scores = result.get(
                "rec_scores",
                []
            )

        except Exception:

            scores = []

        output = []

        for index, text in enumerate(texts):

            clean = clean_plate_text(
                text
            )

            if not clean:
                continue

            score = 0.0

            if index < len(scores):

                try:

                    score = float(
                        scores[index]
                    )

                except Exception:

                    score = 0.0

            output.append(
                (
                    clean,
                    score
                )
            )

        return output

    except Exception as e:

        print(
            "OCR engine error:",
            e
        )

        return []


# ============================================================
# BEST PLATE
# ============================================================

def get_best_plate(results):

    candidates = []

    for text, score in results:

        text = clean_plate_text(
            text
        )

        if not is_valid_plate(text):
            continue

        candidates.append(
            (
                text,
                float(score)
            )
        )

    if not candidates:
        return "", 0.0

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return candidates[0]


# ============================================================
# ============================================================
# CREATE ENHANCED PREPROCESSING VARIANTS
# ============================================================

def create_fallback_image(image):
    """Contrast enhancement with bilateral denoising and CLAHE."""
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def create_threshold_image(image):
    """Adaptive/Otsu threshold variant for challenging illumination and shadows."""
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ============================================================
# READ LICENSE PLATE
# ============================================================

def read_plate(image):
    """
    Multi-variant number plate OCR pipeline:
      1. Scaled raw crop (aspect ratio strictly preserved)
      2. Enhanced contrast variant (bilateral denoise + CLAHE)
      3. Otsu binarized threshold variant
    Selects the highest-confidence valid alphanumeric plate result.
    """
    if image is None:
        print("Invalid plate image.")
        return ""

    if image.size == 0:
        print("Empty plate image.")
        return ""

    try:
        height, width = image.shape[:2]
        print(f"Plate image: {width}x{height}")

        # ----------------------------------------------------
        # Variant 1: Scaled Raw (Aspect Ratio Preserved)
        # ----------------------------------------------------
        fast_image = cv2.resize(
            image,
            None,
            fx=PLATE_SCALE,
            fy=PLATE_SCALE,
            interpolation=cv2.INTER_CUBIC
        )

        results = run_ocr(fast_image)
        text, score = get_best_plate(results)

        if text:
            print(f"Fast OCR: {text} (confidence {score:.2f})")
            if score >= FAST_ACCEPT_SCORE:
                return text

        # ----------------------------------------------------
        # Variant 2: Enhanced Contrast (Bilateral + CLAHE)
        # ----------------------------------------------------
        fallback_image = create_fallback_image(image)
        if fallback_image is not None:
            fallback_scaled = cv2.resize(
                fallback_image,
                None,
                fx=PLATE_SCALE,
                fy=PLATE_SCALE,
                interpolation=cv2.INTER_CUBIC
            )
            results = run_ocr(fallback_scaled)
            fb_text, fb_score = get_best_plate(results)
            if fb_text and fb_score > score:
                text = fb_text
                score = fb_score
                print(f"Fallback CLAHE OCR: {text} (confidence {score:.2f})")
                if score >= FAST_ACCEPT_SCORE:
                    return text

        # ----------------------------------------------------
        # Variant 3: Otsu Thresholding Variant
        # ----------------------------------------------------
        thresh_image = create_threshold_image(image)
        if thresh_image is not None:
            thresh_scaled = cv2.resize(
                thresh_image,
                None,
                fx=PLATE_SCALE,
                fy=PLATE_SCALE,
                interpolation=cv2.INTER_CUBIC
            )
            results = run_ocr(thresh_scaled)
            th_text, th_score = get_best_plate(results)
            if th_text and th_score > score:
                text = th_text
                score = th_score
                print(f"Threshold OCR: {text} (confidence {score:.2f})")

        # ----------------------------------------------------
        # Accept Best Valid Candidate
        # ----------------------------------------------------
        if text and score >= FALLBACK_ACCEPT_SCORE:
            print(f"Detected plate: {text}")
            return text

        print("No reliable number plate detected.")
        return ""

    except Exception as e:

        print(
            "Plate OCR failed:",
            e
        )

        return ""


# ============================================================
# READ BUS BODY NUMBER
# ============================================================

def read_bus_number(image):

    if image is None:
        return ""

    if image.size == 0:
        return ""

    try:

        # ====================================================
        # RESIZE BUS CROP
        # ====================================================

        enlarged = cv2.resize(
            image,
            None,
            fx=BUS_SCALE,
            fy=BUS_SCALE,
            interpolation=cv2.INTER_CUBIC
        )

        results = run_ocr(
            enlarged
        )

        if not results:

            print(
                "Bus number OCR: nothing found."
            )

            return ""

        candidates = []

        for text, score in results:

            raw_text = str(
                text
            ).upper()

            # ------------------------------------------------
            # Keep digits only
            # ------------------------------------------------

            number = re.sub(
                r"[^0-9]",
                "",
                raw_text
            )

            # Current DB bus numbers:
            # 1-4 digits supported.
            if 1 <= len(number) <= 4:

                candidates.append(
                    (
                        number,
                        float(score)
                    )
                )

        if not candidates:

            print(
                "Bus number OCR: NOT READ"
            )

            return ""

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        bus_number = candidates[0][0]

        print(
            "Detected Bus Number:",
            bus_number
        )

        return bus_number

    except Exception as e:

        print(
            "Bus number OCR error:",
            e
        )

        return ""