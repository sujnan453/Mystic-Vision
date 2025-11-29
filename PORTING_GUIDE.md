# Porting Guide: Dr. Strange Shields

This guide outlines the steps to transfer and set up the **Dr. Strange Shields** project on a new laptop or machine.

## 1. Prerequisites

Before you begin, ensure the new machine has the following:

*   **Operating System:** Windows 10/11 (Recommended), macOS, or Linux.
*   **Python:** Version 3.10 or higher is recommended.
*   **Hardware:**
    *   Webcam (Built-in or External).
    *   GPU (Optional but recommended for smoother performance).

## 2. File Transfer

You need to copy the entire project directory to the new machine. Ensure the following critical files and folders are included:

    ```bash
    pip uninstall opencv-python opencv-contrib-python
    pip install opencv-python opencv-contrib-python
    ```
*   **Slow Performance:**
    *   Ensure your laptop is plugged in (power mode).
    *   Reduce the camera resolution in `config.py` (e.g., set `FRAME_WIDTH` and `FRAME_HEIGHT` to lower values).
    *   Use the `--max_hands 1` flag to reduce processing load.

## 6. Verification

To verify the installation:
1.  Run the script.
2.  Perform the "Activation" gesture (Key 1 -> Key 3).
3.  Confirm that the shield effect appears on your hand.

## 7. Project Improvement Suggestions

Here are several suggestions to improve the quality, maintainability, and performance of the project.

### 7.1 Code Quality & Refactoring
*   **Remove Hardcoded Values:** `dataset_collection.py` uses `cv2.VideoCapture(2)`. Use `argparse` to allow the user to specify the camera index (e.g., `--camera 0`).
*   **Reduce Code Duplication:** Refactor `draw_limit_rh` and `draw_limit_lh` in `utils.py` into a single generic function `draw_hand_limit`.
*   **Translate Comments:** Translate Italian comments in `utils.py` to English.

### 7.2 Robustness & Error Handling
*   **Check Camera Availability:** Add a check after `cap.isOpened()` in `dataset_collection.py` to handle connection failures gracefully.

### 7.3 Performance Optimization
*   **Optimize Drawing Specs:** Define `mp_drawing.DrawingSpec` as global constants in `utils.py` to avoid re-initializing them every frame.

### 7.4 User Experience (UX)
*   **Non-Blocking Countdown:** Replace blocking `cv2.waitKey()` calls in `dataset_collection.py` with a non-blocking timer to keep the video feed smooth.

### 7.5 Feature Enhancements
*   **Dynamic Dataset Collection:** Allow users to specify gesture labels via command-line arguments.
*   **Model Versioning:** Automatically append timestamps to model filenames in `train_svm.py` to prevent overwriting.
