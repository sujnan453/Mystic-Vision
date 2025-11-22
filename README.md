
# 🛡️ Dr. Strange Shields – Gesture Magic in Action

**Dr. Strange Shields** is a real-time hand-tracking and gesture-recognition project that lets you cast magical shields using your webcam. The code uses Python, OpenCV, and MediaPipe to detect up to 6 hands, recognize specific gesture sequences with an SVM model, and overlay animated shield effects on your video feed.

## What the Code Does

- **Multi-Hand Detection:** Detects up to 6 hands simultaneously using MediaPipe Hands, with smooth tracking and position smoothing for stable effects.
- **Gesture Recognition:** Uses a trained SVM model to recognize specific hand gesture sequences (e.g., KEY_1 → KEY_3) to activate or deactivate shields.
- **Shield Rendering:** Overlays animated or procedural shield effects on each detected hand in real time, scaling and positioning the effect based on hand location.
- **Flexible Output:** Shows the result in an OpenCV window, a virtual camera, or both, so you can use the effect in video calls or recordings.
- **User Controls:** Includes auto-brightness, UI overlays, and keyboard shortcuts for toggling features and switching effects.

## How to Run

1. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
2. **Run the main script:**
	```bash
	python shield.py --camera 0 --ML_model models/model_svm.sav
	```
	- Use `--camera` to select your webcam index (default is 0).
	- Use `--output_mode` to choose between `window`, `virtual`, or `both`.
	- Use `--max_hands` to set the maximum number of hands to detect (default is 6).
	- Use `--shield_video` to select a custom shield effect.

3. **Make gestures in front of your camera to activate shields!**

## Example Workflow

1. The camera feed is captured and processed in real time.
2. MediaPipe detects all visible hands and tracks their positions.
3. The SVM model recognizes gesture sequences to activate or deactivate shields.
4. Animated shield effects are rendered and overlaid on each detected hand.
5. The output is displayed in a window and/or sent to a virtual camera.

---

<p align="center"> <img width="640" src="./images/example.png"> </p>

---

---

## Gesture Flow

- **Activation:** KEY_1 → KEY_3 → Shields ON
- **Deactivation:** KEY_4 → Shields OFF

<p align="center"> <img width="320" src="./images/position_1.png"> <img width="320" src="./images/position_2.png"> <img width="320" src="./images/position_3.png"> </p> <p align="center"> <img width="360" src="./images/position_4.png"> </p>

## Tips & Tricks

- Keep both hands visible and well-lit for best detection.
- Adjust confidence thresholds if gestures are misread.
- Lower camera resolution for smoother performance on slower machines.

---

Made with Python 🐍, hand gestures 🤲, and a touch of CodeXpert magic ✨
