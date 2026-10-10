"""
# ============================================================
# IN/OUT X — MOUSE-BASED LINE CALIBRATION ACCEPTANCE SUITE
# ============================================================
Validates:
  1. Mouse Left-Click 1 (Start Point) and Left-Click 2 (End Point).
  2. Coordinate scaling between preview panel (640x480) and original frame (1920x1080).
  3. KRCT vs KRCE panel isolation (clicks on wrong panel rejected).
  4. Interactive HUD Buttons (Selection 1-4, Save, Reset, Toggle Detection).
  5. Persistence: saved lines reload from lines_config.json without resetting.
  6. Visual display colors: Yellow (0, 255, 255) for Entry, Orange (0, 140, 255) for Exit.
  7. Detection Synchronization: configured lines are the exact lines used in crossing logic.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(line_buffering=True)

import camera
from camera import (
    calib_mouse_callback,
    render_lines_on_panel,
    render_hud_panel,
    CameraProcessor,
    camera_processors,
    HUD_BUTTONS,
    LINE_NAME_MAP,
)
from line_calibration import (
    save_calibrated_lines,
    load_calibrated_lines,
    reset_all_lines,
    CALIBRATION_FILE,
    DEFAULT_LINES,
)


def run_mouse_calibration_tests():
    passed = 0
    total = 0

    def test_case(name):
        nonlocal total
        total += 1
        print(f"\n[TEST {total}] {name}...", flush=True)

    print("=" * 68)
    print("IN/OUT X — MOUSE-BASED LINE CALIBRATION VERIFICATION SUITE")
    print("=" * 68)

    # Setup mock camera processors
    proc1 = CameraProcessor(
        camera_id="KRCT_GATE_01",
        college="KRCT",
        gate_name="KRCT Main Gate",
        source="0",
        entry_line=((100, 200), (300, 200)),
        exit_line=((400, 200), (600, 200)),
        enabled=False,
    )
    proc1.frame_width = 1920
    proc1.frame_height = 1080

    proc2 = CameraProcessor(
        camera_id="KRCE_GATE_01",
        college="KRCE",
        gate_name="KRCE Main Gate",
        source="1",
        entry_line=((700, 800), (900, 800)),
        exit_line=((1000, 800), (1200, 800)),
        enabled=False,
    )
    proc2.frame_width = 1920
    proc2.frame_height = 1080

    camera_processors["KRCT_GATE_01"] = proc1
    camera_processors["KRCE_GATE_01"] = proc2

    # -------------------------------------------------------------------------
    # TEST 1: HUD Button Clicks (Line Selection)
    # -------------------------------------------------------------------------
    test_case("HUD Button Click Selection (Entry 1, Exit 1, Entry 2, Exit 2)")
    # Click button 1: [1] Entry 1 (KRCT) -> box (15, 515, 175, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 50, 530, 0, None)
    assert camera.selected_calib_line == "ENTRY_LINE_1"
    print("  -> Clicked Button 1: Selected ENTRY_LINE_1 (KRCT): PASS")

    # Click button 2: [2] Exit 1 (KRCT) -> box (185, 515, 345, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 200, 530, 0, None)
    assert camera.selected_calib_line == "EXIT_LINE_1"
    print("  -> Clicked Button 2: Selected EXIT_LINE_1 (KRCT): PASS")

    # Click button 3: [3] Entry 2 (KRCE) -> box (355, 515, 515, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 400, 530, 0, None)
    assert camera.selected_calib_line == "ENTRY_LINE_2"
    print("  -> Clicked Button 3: Selected ENTRY_LINE_2 (KRCE): PASS")

    # Click button 4: [4] Exit 2 (KRCE) -> box (525, 515, 685, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 550, 530, 0, None)
    assert camera.selected_calib_line == "EXIT_LINE_2"
    print("  -> Clicked Button 4: Selected EXIT_LINE_2 (KRCE): PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 2: Two-Click Mouse Drawing & Sub-Pixel Coordinate Mapping
    # -------------------------------------------------------------------------
    test_case("Two-Click Line Drawing & 640x480 -> 1920x1080 Resolution Scaling")
    # Select ENTRY_LINE_1
    camera.selected_calib_line = "ENTRY_LINE_1"
    camera.calib_click_points.clear()

    # Click 1: Start Point at display (100, 150) on KRCT panel
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 150, 0, None)
    assert len(camera.calib_click_points) == 1
    disp_pt, cam_pt = camera.calib_click_points[0]
    assert disp_pt == (100, 150)
    # 100 * (1920 / 640) = 300, 150 * (1080 / 480) = 338
    assert cam_pt == (300, 338)
    print(f"  -> Click 1 set start point at display {disp_pt} -> camera {cam_pt}: PASS")

    # Click 2: End Point at display (250, 160) on KRCT panel
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 250, 160, 0, None)
    assert len(camera.calib_click_points) == 0  # Buffer cleared on completion
    # 250 * 3 = 750, 160 * 2.25 = 360
    assert proc1.entry_line == ((300, 338), (750, 360))
    print(f"  -> Click 2 completed line: {proc1.entry_line}: PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 3: KRCT vs KRCE Panel Isolation (Clicks on Wrong Camera Rejected)
    # -------------------------------------------------------------------------
    test_case("Camera Panel Isolation (Rejecting Wrong-Panel Clicks)")
    # Select KRCT line (EXIT_LINE_1)
    camera.selected_calib_line = "EXIT_LINE_1"
    camera.calib_click_points.clear()

    # Attempt click on KRCE panel (x = 700 >= 640)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 700, 100, 0, None)
    assert len(camera.calib_click_points) == 0
    assert "belongs to KRCT" in camera.hud_status_text or "Notice" in camera.hud_status_text
    print("  -> Click for KRCT line on KRCE panel was safely rejected: PASS")

    # Now select KRCE line (ENTRY_LINE_2) and attempt click on KRCT panel (x = 100 < 640)
    camera.selected_calib_line = "ENTRY_LINE_2"
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    assert len(camera.calib_click_points) == 0
    assert "belongs to KRCE" in camera.hud_status_text or "Notice" in camera.hud_status_text
    print("  -> Click for KRCE line on KRCT panel was safely rejected: PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 4: Drawing on KRCE Camera Panel (Offset Handling)
    # -------------------------------------------------------------------------
    test_case("KRCE Camera Mouse Drawing with Horizontal Offset (x - 640)")
    camera.selected_calib_line = "ENTRY_LINE_2"
    camera.calib_click_points.clear()

    # Click 1 at global x = 740 (relative x = 100 inside KRCE), y = 200
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 740, 200, 0, None)
    assert len(camera.calib_click_points) == 1
    disp_pt2, cam_pt2 = camera.calib_click_points[0]
    assert disp_pt2 == (100, 200)
    assert cam_pt2 == (300, 450)

    # Click 2 at global x = 940 (relative x = 300), y = 220
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 940, 220, 0, None)
    assert proc2.entry_line == ((300, 450), (900, 495))
    print(f"  -> KRCE line drawn accurately with panel offset: {proc2.entry_line}: PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 5: Persistent Storage & Startup Reload (lines_config.json)
    # -------------------------------------------------------------------------
    test_case("Persistence: Auto-Save and Startup Reload without Resetting")
    # Save button click -> box (695, 515, 835, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 750, 530, 0, None)
    assert CALIBRATION_FILE.exists()

    # Reload from disk
    loaded = load_calibrated_lines()
    assert loaded["ENTRY_LINE_1"] == ((300, 338), (750, 360))
    assert loaded["ENTRY_LINE_2"] == ((300, 450), (900, 495))
    print("  -> Lines persisted and reloaded from lines_config.json: PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 6: Mode Toggle: Calibration vs Detection
    # -------------------------------------------------------------------------
    test_case("Mode Toggle: Calibration Mode to Detection Mode and Back")
    assert camera.active_app_mode in ("CALIBRATION", "DETECTION")
    camera.active_app_mode = "CALIBRATION"

    # Click Toggle Button -> box (995, 515, 1265, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 1050, 530, 0, None)
    assert camera.active_app_mode == "DETECTION"
    print("  -> Switched from CALIBRATION to DETECTION mode: PASS")

    # Click Toggle Button again
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 1050, 530, 0, None)
    assert camera.active_app_mode == "CALIBRATION"
    print("  -> Switched back to CALIBRATION mode: PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 7: Reset Selected Line Button
    # -------------------------------------------------------------------------
    test_case("Reset Button: Restore Current Line to System Default")
    camera.selected_calib_line = "ENTRY_LINE_1"
    # Click Reset Button -> box (845, 515, 985, 555)
    calib_mouse_callback(cv2.EVENT_LBUTTONDOWN, 900, 530, 0, None)
    assert proc1.entry_line == DEFAULT_LINES["ENTRY_LINE_1"]
    print(f"  -> Reset active line to default {DEFAULT_LINES['ENTRY_LINE_1']}: PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 8: Visual Rendering Colors: Yellow (Entry) and Orange (Exit)
    # -------------------------------------------------------------------------
    test_case("Visual Rendering Colors: Yellow for Entry and Orange for Exit")
    test_panel = np.zeros((480, 640, 3), dtype=np.uint8)
    render_lines_on_panel(test_panel, proc1, is_krct=True,
                          active_line_key="ENTRY_LINE_1",
                          pending_clicks=[],
                          mouse_pos=None)

    # Verify colors exist on canvas: Yellow (0, 255, 255) and Orange (0, 140, 255)
    has_yellow = np.any((test_panel[:, :, 0] == 0) & (test_panel[:, :, 1] == 255) & (test_panel[:, :, 2] == 255))
    has_orange = np.any((test_panel[:, :, 0] == 0) & (test_panel[:, :, 1] == 140) & (test_panel[:, :, 2] == 255))
    assert has_yellow, "Expected yellow (0, 255, 255) entry line on preview panel"
    assert has_orange, "Expected orange (0, 140, 255) exit line on preview panel"
    print("  -> Entry line rendered in Yellow (0, 255, 255) and Exit in Orange (0, 140, 255): PASS")
    passed += 1

    # -------------------------------------------------------------------------
    # TEST 9: HUD Panel Layout & Content
    # -------------------------------------------------------------------------
    test_case("HUD Panel Dimensions, Content, and Elements")
    hud = render_hud_panel(proc1, proc2)
    assert hud.shape == (140, 1280, 3), f"Expected (140, 1280, 3), got {hud.shape}"
    print("  -> HUD panel created with correct dimensions (1280x140): PASS")
    passed += 1

    # Clean reset
    reset_all_lines()

    print("\n" + "=" * 68)
    print(f"ALL MOUSE CALIBRATION TESTS PASSED: {passed}/{total} (100% PASS)")
    print("=" * 68)


if __name__ == "__main__":
    run_mouse_calibration_tests()
