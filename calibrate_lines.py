"""
# ============================================================
# IN/OUT X — VIRTUAL LINE CALIBRATION TOOL
# ============================================================
Calibrate ENTRY and EXIT virtual lines independently for:
  - Camera 1 (KRCT)
  - Camera 2 (KRCE)

Usage:
  python calibrate_lines.py --camera 1
  python calibrate_lines.py --camera 2
"""

import cv2
import sys
import argparse
from config import (
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
    mask_rtsp_url,
)

points = []
selection_done = False


def mouse_callback(event, x, y, flags, param):
    global points, selection_done

    if selection_done:
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) >= 4:
        return

    points.append((x, y))
    pt_num = len(points)

    labels = {
        1: "EXIT Start Point",
        2: "EXIT End Point (Line 1 Completed)",
        3: "ENTRY Start Point",
        4: "ENTRY End Point (Line 2 Completed)"
    }

    print(f"Click {pt_num}: ({x}, {y}) -> {labels.get(pt_num)}")

    if pt_num == 4:
        selection_done = True
        exit_line = (points[0], points[1])
        entry_line = (points[2], points[3])

        print("\n" + "=" * 60)
        print(f"CALIBRATION COMPLETE: {param.get('name')}")
        print("=" * 60)
        print("Copy and paste these coordinates into config.py:")
        print()
        if param.get("num") == 1:
            print(f"EXIT_LINE_1 = {exit_line}")
            print(f"ENTRY_LINE_1 = {entry_line}")
        else:
            print(f"EXIT_LINE_2 = {exit_line}")
            print(f"ENTRY_LINE_2 = {entry_line}")
        print("=" * 60)
        print("Press 'R' to reset and retry.")
        print("Press 'Q' to save and exit.\n")


def calibrate(camera_num=1):
    global points, selection_done

    if camera_num == 1:
        cam_id = CAMERA_1_ID
        college = COLLEGE_1
        gate = GATE_NAME_1
        source = CAMERA_SOURCE_1
        current_exit = EXIT_LINE_1
        current_entry = ENTRY_LINE_1
    else:
        cam_id = CAMERA_2_ID
        college = COLLEGE_2
        gate = GATE_NAME_2
        source = CAMERA_SOURCE_2
        current_exit = EXIT_LINE_2
        current_entry = ENTRY_LINE_2

    param = {"num": camera_num, "name": f"{college} ({cam_id})"}

    print("=" * 65)
    print(f"LINE CALIBRATION — CAMERA {camera_num}: {college} ({cam_id})")
    print(f"Gate   : {gate}")
    print(f"Source : {mask_rtsp_url(source)}")
    print("=" * 65)

    cam_source = source
    if isinstance(cam_source, str) and cam_source.strip().isdigit():
        cam_source = int(cam_source.strip())

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"\n[ERROR] Unable to open stream for Camera {camera_num}: {mask_rtsp_url(source)}")
        print("Generating a test canvas to practice calibration...")
        ret = False
        frame = None
    else:
        ret, frame = cap.read()

    win_name = f"IN/OUT X Calibration - {college} ({cam_id})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback, param)

    print("\n[INSTRUCTIONS]")
    print("Click 4 points in sequence on the video frame:")
    print("  1. EXIT line START")
    print("  2. EXIT line END")
    print("  3. ENTRY line START")
    print("  4. ENTRY line END")
    print("Press 'R' to reset clicks. Press 'Q' to quit.\n")

    while True:
        if cap.isOpened():
            ret, live_frame = cap.read()
            if ret and live_frame is not None:
                frame = live_frame.copy()

        if frame is None:
            # Placeholder canvas
            frame = (cv2.imread("captured_frames/sample.jpg")
                     if cv2.haveImageReader("captured_frames/sample.jpg")
                     else None)
            if frame is None:
                import numpy as np
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw existing lines if any and no clicks yet
        if not points:
            if current_exit:
                cv2.line(display, current_exit[0], current_exit[1], (0, 165, 255), 2)
                cv2.putText(display, "Current EXIT", (current_exit[0][0] + 5, current_exit[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            if current_entry:
                cv2.line(display, current_entry[0], current_entry[1], (0, 255, 255), 2)
                cv2.putText(display, "Current ENTRY", (current_entry[0][0] + 5, current_entry[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw active clicks
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 6, (255, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if len(points) >= 2:
            cv2.line(display, points[0], points[1], (0, 165, 255), 3)
            cv2.putText(display, "NEW EXIT", (points[0][0] + 5, points[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        if len(points) >= 4:
            cv2.line(display, points[2], points[3], (0, 255, 255), 3)
            cv2.putText(display, "NEW ENTRY", (points[2][0] + 5, points[2][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Banner
        cv2.rectangle(display, (0, 0), (w, 35), (20, 20, 25), -1)
        info = f"{college} ({cam_id}) | Resolution: {w}x{h} | Clicks: {len(points)}/4"
        cv2.putText(display, info, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            points.clear()
            selection_done = False
            print("[CALIBRATION] Points reset. Click 4 points again.")

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate IN/OUT X virtual lines")
    parser.add_argument("--camera", type=int, choices=[1, 2], default=1,
                        help="Camera number (1 for KRCT, 2 for KRCE)")
    args = parser.parse_args()
    calibrate(args.camera)
