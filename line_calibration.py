# ============================================================
# IN/OUT X — INTERACTIVE TERMINAL LINE CALIBRATION
# ============================================================

import os
import sys
import json
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_FILE = BASE_DIR / "lines_config.json"

# Default fallback line coordinates
DEFAULT_LINES = {
    "ENTRY_LINE_1": ((157, 635), (145, 727)),
    "EXIT_LINE_1": ((600, 389), (1327, 246)),
    "ENTRY_LINE_2": ((150, 600), (150, 700)),
    "EXIT_LINE_2": ((550, 380), (1200, 250)),
}

LINE_DESCRIPTIONS = {
    "ENTRY_LINE_1": "Entry Line 1 (KRCT)",
    "EXIT_LINE_1": "Exit Line 1 (KRCT)",
    "ENTRY_LINE_2": "Entry Line 2 (KRCE)",
    "EXIT_LINE_2": "Exit Line 2 (KRCE)",
}

# In-memory active coordinates
_current_lines = dict(DEFAULT_LINES)
_lines_lock = threading.Lock()


# ============================================================
# PERSISTENCE (LOAD / SAVE / RESET)
# ============================================================

def _normalize_line_tuple(raw_coords) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Validate and convert raw JSON data to ((int, int), (int, int))."""
    try:
        if not raw_coords or len(raw_coords) != 2:
            return None
        p1, p2 = raw_coords
        if len(p1) != 2 or len(p2) != 2:
            return None
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return None
        if x1 == x2 and y1 == y2:
            return None
        return ((x1, y1), (x2, y2))
    except (ValueError, TypeError, IndexError):
        return None


def load_calibrated_lines() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """
    Load calibrated lines from lines_config.json if present.
    Does not overwrite valid saved coordinates on startup.
    Falls back to defaults for any missing/corrupt keys.
    """
    global _current_lines
    loaded = dict(DEFAULT_LINES)

    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for key in DEFAULT_LINES:
                    if key in data:
                        norm = _normalize_line_tuple(data[key])
                        if norm is not None:
                            loaded[key] = norm
                        else:
                            print(f"[CALIBRATION] Invalid coordinates for {key} in {CALIBRATION_FILE.name}; using default.")
        except Exception as e:
            print(f"[CALIBRATION] Warning: Could not read {CALIBRATION_FILE.name}: {e}")
    else:
        print(f"[CALIBRATION NOTICE] No saved coordinates found in {CALIBRATION_FILE.name}.")
        print("[CALIBRATION NOTICE] Calibration is required before processing cameras.")

    with _lines_lock:
        _current_lines = dict(loaded)

    return dict(loaded)


def save_calibrated_lines(lines_dict: dict | None = None) -> bool:
    """
    Persist current line coordinates to lines_config.json.
    Updates internal memory and writes validated coordinates to disk.
    """
    global _current_lines
    with _lines_lock:
        if lines_dict:
            _current_lines.update(lines_dict)
        to_save = dict(_current_lines)

    # Format as JSON-serializable structure: {"ENTRY_LINE_1": [[x1, y1], [x2, y2]], ...}
    serializable = {}
    for k, v in to_save.items():
        if v and len(v) == 2:
            serializable[k] = [[int(v[0][0]), int(v[0][1])], [int(v[1][0]), int(v[1][1])]]

    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4)
        print(f"[CALIBRATION] Coordinates successfully saved to {CALIBRATION_FILE.name}")
        return True
    except Exception as e:
        print(f"[CALIBRATION ERROR] Failed to save {CALIBRATION_FILE.name}: {e}")
        return False


def reset_all_lines() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """
    Reset all lines to default values and update persistent storage.
    """
    global _current_lines
    with _lines_lock:
        _current_lines = dict(DEFAULT_LINES)
    save_calibrated_lines(_current_lines)
    apply_lines_to_processors()
    print("[CALIBRATION] All lines have been reset to system defaults.")
    return dict(_current_lines)


def get_current_lines() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    with _lines_lock:
        return dict(_current_lines)


# ============================================================
# COORDINATE INPUT & VALIDATION
# ============================================================

def validate_coordinates(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int | None = 1920,
    frame_h: int | None = 1080
) -> tuple[bool, str]:
    """
    Validates integer pixel coordinates:
      - Must be non-negative
      - Endpoints cannot be identical
      - Must not exceed frame dimensions (if known)
    """
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False, "Coordinates cannot be negative."

    if x1 == x2 and y1 == y2:
        return False, "Endpoints cannot be identical (zero-length line)."

    if frame_w is not None and (x1 > frame_w or x2 > frame_w):
        return False, f"X coordinate exceeds frame width ({frame_w}px)."

    if frame_h is not None and (y1 > frame_h or y2 > frame_h):
        return False, f"Y coordinate exceeds frame height ({frame_h}px)."

    return True, ""


def prompt_edit_line(
    line_key: str,
    frame_w: int | None = 1920,
    frame_h: int | None = 1080
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Prompts the user for X1, Y1, X2, Y2 in the exact format required.
    Validates integer pixel inputs and displays confirmation.
    """
    line_label = LINE_DESCRIPTIONS.get(line_key, line_key)
    with _lines_lock:
        current = _current_lines.get(line_key, DEFAULT_LINES[line_key])

    print()
    print(f"Select {line_label}\n")

    def _read_int(prompt: str) -> int | None:
        while True:
            raw = input(prompt).strip()
            if not raw:
                return None
            try:
                val = int(raw)
                return val
            except ValueError:
                print("  [ERROR] Invalid input. Please enter a whole integer pixel value.")

    x1 = _read_int("Enter X1: ")
    if x1 is None:
        print(f"Edit cancelled. Kept current coordinates: {current}")
        return current

    y1 = _read_int("Enter Y1: ")
    if y1 is None:
        print(f"Edit cancelled. Kept current coordinates: {current}")
        return current

    x2 = _read_int("Enter X2: ")
    if x2 is None:
        print(f"Edit cancelled. Kept current coordinates: {current}")
        return current

    y2 = _read_int("Enter Y2: ")
    if y2 is None:
        print(f"Edit cancelled. Kept current coordinates: {current}")
        return current

    is_valid, err_msg = validate_coordinates(x1, y1, x2, y2, frame_w, frame_h)
    if not is_valid:
        print(f"\n[VALIDATION REJECTED] {err_msg}")
        print(f"Coordinates were not changed. Kept: {current}\n")
        return current

    new_line = ((x1, y1), (x2, y2))
    with _lines_lock:
        _current_lines[line_key] = new_line

    print()
    print(f"{line_label} updated: {new_line}")
    apply_lines_to_processors()
    return new_line


def view_current_coordinates():
    with _lines_lock:
        lines = dict(_current_lines)

    print()
    print("=" * 52)
    print("          CURRENT LINE COORDINATES")
    print("=" * 52)
    print(f"1. Entry Line 1 (KRCT) : {lines['ENTRY_LINE_1']}")
    print(f"2. Exit Line 1 (KRCT)  : {lines['EXIT_LINE_1']}")
    print(f"3. Entry Line 2 (KRCE) : {lines['ENTRY_LINE_2']}")
    print(f"4. Exit Line 2 (KRCE)  : {lines['EXIT_LINE_2']}")
    print("=" * 52)
    print()


# ============================================================
# INTERACTIVE TERMINAL MENU
# ============================================================

def terminal_calibration_menu(auto_start: bool = False, is_runtime: bool = False) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """
    Displays the interactive line calibration menu at application start or runtime.
    If stdin is not an interactive terminal (e.g. piped or automated test),
    loads saved coordinates and proceeds immediately.
    """
    load_calibrated_lines()

    # Bypass interactive prompts in headless / piped environments
    if auto_start or not sys.stdin.isatty():
        print("[CALIBRATION] Non-interactive environment detected. Loaded active coordinates:")
        view_current_coordinates()
        return get_current_lines()

    while True:
        print()
        print("====================================================")
        print("          IN/OUT X — LINE CALIBRATION")
        print("====================================================")
        print()
        print("1. Edit Entry Line 1 (KRCT)")
        print("2. Edit Exit Line 1 (KRCT)")
        print("3. Edit Entry Line 2 (KRCE)")
        print("4. Edit Exit Line 2 (KRCE)")
        print("5. View Current Line Coordinates")
        print("6. Reset All Lines")
        print("7. Save Coordinates")
        if is_runtime:
            print("8. Return to Camera Processing")
            print("9. Return to Camera Processing")
        else:
            print("8. Start Camera Processing")
            print("9. Exit Application")
        print()

        try:
            choice = input("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            if is_runtime:
                return get_current_lines()
            print("\nExiting Application.")
            sys.exit(0)

        if choice == "1":
            prompt_edit_line("ENTRY_LINE_1")
        elif choice == "2":
            prompt_edit_line("EXIT_LINE_1")
        elif choice == "3":
            prompt_edit_line("ENTRY_LINE_2")
        elif choice == "4":
            prompt_edit_line("EXIT_LINE_2")
        elif choice == "5":
            view_current_coordinates()
        elif choice == "6":
            confirm = input("Are you sure you want to reset all 4 lines to defaults? (y/n): ").strip().lower()
            if confirm in ("y", "yes"):
                reset_all_lines()
                view_current_coordinates()
            else:
                print("Reset cancelled.")
        elif choice == "7":
            save_calibrated_lines()
            view_current_coordinates()
        elif choice == "8":
            save_calibrated_lines()
            apply_lines_to_processors()
            if is_runtime:
                print("\nReturning to live Camera Processing with updated coordinates...")
            else:
                print("\nStarting Camera Processing with current coordinates...")
            view_current_coordinates()
            return get_current_lines()
        elif choice == "9":
            if is_runtime:
                print("\nReturning to live Camera Processing...")
                return get_current_lines()
            print("\nExiting Application. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option. Please choose a number from 1 to 9.")


# ============================================================
# RUNTIME EDITING (WHILE CAMERAS RUN)
# ============================================================

def apply_lines_to_processors():
    """Dynamically applies active coordinates to running CameraProcessors."""
    try:
        from camera import get_processor
        from config import CAMERA_1_ID, CAMERA_2_ID

        with _lines_lock:
            lines = dict(_current_lines)

        proc1 = get_processor(CAMERA_1_ID)
        if proc1:
            proc1.entry_line = lines["ENTRY_LINE_1"]
            proc1.exit_line = lines["EXIT_LINE_1"]

        proc2 = get_processor(CAMERA_2_ID)
        if proc2:
            proc2.entry_line = lines["ENTRY_LINE_2"]
            proc2.exit_line = lines["EXIT_LINE_2"]

        print(f"[CALIBRATION] Runtime update applied to active camera processors.")
    except Exception as e:
        print(f"[CALIBRATION] Notice: Could not push to running processors ({e})")


def start_runtime_terminal_listener():
    """
    Spawns a background daemon thread that allows the operator to
    type 'c' + Enter into the terminal while dual cameras are streaming,
    allowing live runtime line editing without blocking frame processing.
    """
    if not sys.stdin.isatty():
        return None

    def _listener():
        print("\n[TERMINAL CONTROLS] Type 'c' + Enter in terminal anytime to edit lines live, or 'v' to view:")
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd == "c":
                    print("\n[PAUSING TERMINAL FOR LIVE CALIBRATION]")
                    terminal_calibration_menu(auto_start=False, is_runtime=True)
                    print("\n[LIVE CAMERAS RUNNING] New lines active in real time!")
                    print("[TERMINAL CONTROLS] Type 'c' + Enter to edit lines live, or 'v' to view:")
                elif cmd == "v":
                    view_current_coordinates()
                    print("[TERMINAL CONTROLS] Type 'c' + Enter to edit lines live, or 'v' to view:")
            except Exception:
                break

    t = threading.Thread(target=_listener, daemon=True, name="TerminalLineListener")
    t.start()
    return t

