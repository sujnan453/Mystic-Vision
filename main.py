import cv2
import time
import mediapipe as mp
import numpy as np
import os
import sys
import signal
import threading
from argparse import ArgumentParser

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:
    pyvirtualcam = None
    PixelFormat = None

# Import new modules
import state
from state import update_init_status, init_status, init_lock
from camera import CameraThread
import ui
import effects
import inference
from utils import points_detection_hands
from config import config

# -------------------- Argument Parsing --------------------
parser = ArgumentParser(description="Mystic Vision - Exhibition Mode")
parser.add_argument('--camera', type=int, default=config.CAMERA_ID, help=f'Camera index (default: {config.CAMERA_ID})')
parser.add_argument('--ML_model', type=str, default=config.MODEL_PATH, help='Path to ML model file')
parser.add_argument('--shield_video', type=str, default='effects/Eldritch Mandala.mp4', help='Path to shield video or effect')
parser.add_argument('--output_mode', type=str, default='window', choices=['window', 'virtual', 'both'], help='Output mode: window, virtual, or both')
parser.add_argument('--demo_mode', action='store_true', help='Enable demo mode (easier for kids)')
parser.add_argument('--mirror', action='store_false', help='Do not mirror camera image (default: no mirror)')
parser.add_argument('--buffer_seconds', type=float, default=config.EFFECT_BUFFER_SECONDS, help='Buffer seconds for video effects')
parser.add_argument('--preload_count', type=int, default=config.PRELOAD_COUNT, help='Number of effects to preload')
parser.add_argument('--start_fullscreen', action='store_true', help='Start in fullscreen mode')
parser.add_argument('--gesture_mode', action='store_true', help='Enable gesture mode (require gestures to activate shields)')
parser.add_argument('--no_enhance', action='store_true', help='Disable image enhancement')
parser.add_argument('--pred_every_n', type=int, default=config.PRED_EVERY_N_FRAMES, help='Predict every N frames')
parser.add_argument('--mp_every_n', type=int, default=config.MP_EVERY_N_FRAMES, help='Run MediaPipe every N frames')
parser.add_argument('--min_detection_confidence', type=float, default=config.MIN_DETECTION_CONFIDENCE, help='MediaPipe min detection confidence')
parser.add_argument('--min_tracking_confidence', type=float, default=config.MIN_TRACKING_CONFIDENCE, help='MediaPipe min tracking confidence')
parser.add_argument('--max_hands', type=int, default=config.MAX_HANDS, help='Maximum number of hands to detect')
args = parser.parse_args()

current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# -------------------- Signal Handler --------------------
def signal_handler(sig, frame):
    print("\n\n" + "="*60 + "\n🛑 Interruption received (Ctrl+C)\n🧹 Cleaning resources...")
    state.stop_threads = True
    # Only release resources that exist
    objs = []
    if 'camera_thread' in globals():
        objs.append(camera_thread)
    if 'cam' in globals() and cam is not None:
        objs.append(cam)
    objs += list(effects.shield_loaders.values())
    for obj in objs:
        try:
            if hasattr(obj, 'release'):
                obj.release()
            elif hasattr(obj, 'close'):
                obj.close()
            elif hasattr(obj, 'stop'):
                obj.stop()
        except Exception:
            pass
    try:
        if show_window: cv2.destroyAllWindows()
    except: pass
    print("\n🏁 Application terminated\n" + "="*60)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -------------------- Initialization --------------------
# UI Controls
brightness_boost = 20
show_controls = config.SHOW_CONTROLS
auto_brightness_enabled = True
last_auto_brightness_time = 0

# Tutorial and Exhibition Mode
tutorial_enabled = config.TUTORIAL_ENABLED
show_help_overlay = False
last_activity_time = None
AUTO_RESET_SECONDS = config.AUTO_RESET_SECONDS
session_state = "WELCOME"

# Auto-Switching Configuration
AUTO_SWITCH_ENABLED = config.AUTO_SWITCH_ENABLED
EFFECT_CYCLE_DURATION = config.EFFECT_CYCLE_DURATION
COUNTDOWN_DURATION = config.COUNTDOWN_DURATION
shield_active_start_time = None

# Animation Name Display
effect_name_opacity = 0.0
last_effect_switch_time = None
EFFECT_NAME_FADE_IN_DURATION = 0.5
EFFECT_NAME_DISPLAY_DURATION = 3.0
EFFECT_NAME_FADE_OUT_DURATION = 0.5

# Start camera
update_init_status("Initializing...", 5)
camera_thread = CameraThread(args.camera)
camera_thread.start()
width = camera_thread.width
height = camera_thread.height

# Initialize effects
effects.initialize_procedural_effects(width, height)
update_init_status("Loading ML model...", 30)

# Load model
model_path = args.ML_model if os.path.isabs(args.ML_model) else os.path.join(current_directory, args.ML_model)
model = inference.load_model(model_path)
update_init_status("Warming up model...", 35)

# Warm up model
try:
    dummy_len = 63
    try:
        dummy_len = len(points_detection_hands(np.zeros((21,3)))) if callable(points_detection_hands) else 63
    except Exception:
        dummy_len = 63
    dummy = np.zeros((1, dummy_len), dtype=float)
except Exception:
    dummy = np.zeros((1, 63))

if model:
    for _ in range(3):
        try:
            _ = model.predict_proba(dummy)
        except Exception:
            try:
                _ = model.predict(dummy)
            except Exception:
                break

update_init_status("Model ready", 40)

# State flags
KEY_1 = False
KEY_3 = False
SHIELDS = False
scale = 1.5
mp_hands = mp.solutions.hands

update_init_status("Loading effects...", 45)

# Effects setup
effects_folder = config.EFFECTS_DIR
effects_files = effects.list_effects_files(effects_folder)

shield_path = args.shield_video
if not os.path.isabs(shield_path):
    shield_path = os.path.join(current_directory, shield_path)
shield_loaded_name = effects_files[0] if effects_files else 'Rotating Mandala'

# Preload effects
PRELOAD_FIRST_N = max(1, min(args.preload_count, len(effects_files)))
for i, fname in enumerate(effects_files[:PRELOAD_FIRST_N]):
    progress = 55 + int((i / PRELOAD_FIRST_N) * 30)
    update_init_status(f"Preloading effect {i+1}/{PRELOAD_FIRST_N}...", progress)
    effects.start_loader_for_filename(fname, effects_folder, args.buffer_seconds)
    time.sleep(0.3)

if shield_loaded_name and os.path.basename(shield_loaded_name) in effects_files:
    effects.start_loader_for_filename(os.path.basename(shield_loaded_name), effects_folder, args.buffer_seconds)

update_init_status("Finalizing setup...", 90)

# Start inference thread
inf_thread = inference.start_inference_thread(model)

# Demo mode settings
if args.demo_mode:
    PRED_CONF_THRESHOLD = 0.65
    KEY3_REQUIRED_IN_WINDOW = 2
    KEY3_WINDOW_FRAMES = 4
    KEY_SEQUENCE_TIME = 2.5
else:
    PRED_CONF_THRESHOLD = 0.72
    KEY3_REQUIRED_IN_WINDOW = 3
    KEY3_WINDOW_FRAMES = 5
    KEY_SEQUENCE_TIME = 1.5

PRED_EVERY_N_FRAMES = max(1, args.pred_every_n)
MP_EVERY_N_FRAMES = max(1, args.mp_every_n)

_recent_preds = []
_frame_count = 0
_mp_frame_count = 0

# UI elements
BUTTON_W, BUTTON_H = config.BUTTON_WIDTH, config.BUTTON_HEIGHT
BUTTON_MARGIN = config.BUTTON_MARGIN
button_tl = (width - BUTTON_W - BUTTON_MARGIN, height - BUTTON_H - BUTTON_MARGIN)
button_br = (width - BUTTON_MARGIN, height - BUTTON_MARGIN)

SELECT_BTN_W, SELECT_BTN_H = config.SELECT_BTN_WIDTH, config.SELECT_BTN_HEIGHT
SELECT_BTN_MARGIN = config.SELECT_BTN_MARGIN
select_btn_tl = (width - SELECT_BTN_W - SELECT_BTN_MARGIN, SELECT_BTN_MARGIN)
select_btn_br = (width - SELECT_BTN_MARGIN, SELECT_BTN_MARGIN + SELECT_BTN_H)

menu_open = False
selected_index = 0
if effects_files:
    if shield_loaded_name and shield_loaded_name in effects_files:
        selected_index = effects_files.index(shield_loaded_name)

# Mouse callback
mouse_clicked = False
def on_mouse(event, x, y, flags, param):
    global SHIELDS, mouse_clicked, menu_open, selected_index, shield_loaded_name
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1 = button_tl
        bx2, by2 = button_br
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            SHIELDS = not SHIELDS
            mouse_clicked = True
            return
        
        sx1, sy1 = select_btn_tl
        sx2, sy2 = select_btn_br
        if sx1 <= x <= sx2 and sy1 <= y <= sy2:
            menu_open = not menu_open
            mouse_clicked = True
            return
        
        if menu_open:
            menu_x = sx1
            menu_y = sy2 + 8
            padding = 6
            item_h = 36
            max_items_display = min(12, max(1, len(effects_files)))
            menu_w = SELECT_BTN_W
            menu_h = padding + item_h * max_items_display + padding
            if menu_x <= x <= menu_x + menu_w and menu_y <= y <= menu_y + menu_h:
                rel_y = y - menu_y - padding
                if rel_y < 0:
                    return
                idx = int(rel_y // item_h)
                if 0 <= idx < len(effects_files):
                    chosen = effects_files[idx]
                    selected_index = idx
                    shield_loaded_name = chosen
                    effects.start_loader_for_filename(chosen, effects_folder, args.buffer_seconds)
                    print(f"\n🔁 Selected shield file: {chosen}")
                    menu_open = False
                    mouse_clicked = True
                    return
            else:
                menu_open = False
                mouse_clicked = True
                return

show_window = args.output_mode in ['window', 'both']
use_virtual_cam = args.output_mode in ['virtual', 'both']

def on_brightness_change(val):
    global brightness_boost
    brightness_boost = val

# -------------------- Loading Screen Display --------------------
update_init_status("Preparing display...", 95)

if show_window:
    cv2.namedWindow("Mystic Vision", cv2.WINDOW_NORMAL)
    if args.start_fullscreen or config.FULLSCREEN:
        cv2.setWindowProperty("Mystic Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty("Mystic Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow("Mystic Vision", width, height)
        except Exception:
            pass
    
    loading_start = time.time()
    while time.time() - loading_start < 2.0:
        with init_lock:
            current_step = init_status["step"]
            current_progress = init_status["progress"]
        
        loading_frame = ui.create_loading_screen(width, height, current_step, current_progress)
        cv2.imshow("Mystic Vision", loading_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            state.stop_threads = True
            break
        
        if current_progress < 98:
            update_init_status(current_step, current_progress + 1)

update_init_status("Ready!", 100)

if show_window:
    final_loading = ui.create_loading_screen(width, height, "System Ready", 100)
    cv2.waitKey(500)

# -------------------- Main loop --------------------
cached_mp_results = None
cached_mp_frame = None
target_fps = 60
frame_time = 1.0 / target_fps
last_frame_time = time.time()

with mp_hands.Hands(
    min_detection_confidence=args.min_detection_confidence,
    min_tracking_confidence=args.min_tracking_confidence,
    model_complexity=1,
    max_num_hands=args.max_hands
) as hands:

    if use_virtual_cam:
        cam = pyvirtualcam.Camera(width, height, target_fps, fmt=PixelFormat.BGR)
        print(f"Virtual cam: {cam.device}")
    else:
        cam = None

    if show_window:
        cv2.setMouseCallback("Mystic Vision", on_mouse)
        cv2.createTrackbar('Brightness', 'Mystic Vision', brightness_boost, 100, on_brightness_change)

    try:
        while camera_thread.running and not state.stop_threads:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            ret, frame = camera_thread.read()
            if not ret or frame is None:
                time.sleep(0.008)
                continue

            frame = np.ascontiguousarray(frame)

            if args.mirror:
                frame = cv2.flip(frame, 1)
            
            if not args.no_enhance:
                if auto_brightness_enabled:
                    brightness_boost, last_auto_brightness_time = ui.auto_adjust_brightness(frame, brightness_boost, last_auto_brightness_time)
                
                frame = ui.enhance_frame(frame, brightness_boost)
                frame = np.ascontiguousarray(frame)

            _frame_count += 1
            _mp_frame_count += 1

            current_shield_name = shield_loaded_name if shield_loaded_name else (effects_files[selected_index] if effects_files else None)
            frame_shield, shield_ok = effects.get_shield_frame(current_shield_name)
            if frame_shield is None:
                frame_shield = np.zeros((200,200,3), dtype=np.uint8)
            else:
                frame_shield = np.ascontiguousarray(frame_shield)

            if _mp_frame_count >= MP_EVERY_N_FRAMES or cached_mp_results is None:
                frame.flags.writeable = False
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                frame.flags.writeable = True
                
                cached_mp_results = results
                cached_mp_frame = frame.copy()
                _mp_frame_count = 0
            else:
                results = cached_mp_results

            SMOOTHING_FACTOR = 0.7
            if not hasattr(globals(), 'smoothed_hand_centers'):
                smoothed_hand_centers = {}
            else:
                smoothed_hand_centers = globals().get('smoothed_hand_centers', {})

            hand_centers = []
            if results and results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    x_list = [landmark.x for landmark in hand_landmarks.landmark]
                    y_list = [landmark.y for landmark in hand_landmarks.landmark]
                    xMin = min(x_list)
                    xMax = max(x_list)
                    yMin = min(y_list)
                    yMax = max(y_list)

                    center_x = (xMax + xMin) / 2
                    center_y = (yMax + yMin) / 2
                    hand_id = f"hand_{idx}"
                    prev = smoothed_hand_centers.get(hand_id, (center_x, center_y, xMin, xMax, yMin, yMax))
                    smooth_x = SMOOTHING_FACTOR * prev[0] + (1 - SMOOTHING_FACTOR) * center_x
                    smooth_y = SMOOTHING_FACTOR * prev[1] + (1 - SMOOTHING_FACTOR) * center_y
                    smooth_xMin = SMOOTHING_FACTOR * prev[2] + (1 - SMOOTHING_FACTOR) * xMin
                    smooth_xMax = SMOOTHING_FACTOR * prev[3] + (1 - SMOOTHING_FACTOR) * xMax
                    smooth_yMin = SMOOTHING_FACTOR * prev[4] + (1 - SMOOTHING_FACTOR) * yMin
                    smooth_yMax = SMOOTHING_FACTOR * prev[5] + (1 - SMOOTHING_FACTOR) * yMax
                    smoothed_hand_centers[hand_id] = (smooth_x, smooth_y, smooth_xMin, smooth_xMax, smooth_yMin, smooth_yMax)
                    hand_centers.append((smooth_xMin, smooth_xMax, smooth_yMin, smooth_yMax, hand_landmarks))

                if SHIELDS:
                    mask = cv2.inRange(frame_shield, np.array([0,0,0]), np.array([0,0,0]))
                    res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
                    res = frame_shield - res
                    alpha = 1

                    for idx, (smooth_xMin, smooth_xMax, smooth_yMin, smooth_yMax, hand_landmarks) in enumerate(hand_centers):
                        hand_id = f"hand_{idx}"
                        smooth_x, smooth_y = smoothed_hand_centers[hand_id][0], smoothed_hand_centers[hand_id][1]
                        xc = int(width * smooth_x)
                        yc = int(height * smooth_y)
                        w_shield = int(width * (smooth_xMax - smooth_xMin) / 2 * 3.5 * scale)
                        h_shield = int(height * (smooth_yMax - smooth_yMin) / 2 * 2 * scale)
                        if w_shield > 0 and h_shield > 0:
                            res_hand = cv2.resize(res, (max(1, w_shield * 2), max(1, h_shield * 2)), interpolation=cv2.INTER_CUBIC)
                            start_h = 0; start_w = 0
                            stop_h = h_shield * 2; stop_w = w_shield * 2
                            f_start_h = yc - h_shield; f_stop_h = yc + h_shield
                            f_start_w = xc - w_shield; f_stop_w = xc + w_shield
                            if f_start_h < 0:
                                start_h = -f_start_h; f_start_h = 0
                            if f_stop_h > height:
                                stop_h = stop_h - (f_stop_h - height); f_stop_h = height
                            if f_start_w < 0:
                                start_w = -f_start_w; f_start_w = 0
                            if f_stop_w > width:
                                stop_w = stop_w - (f_stop_w - width); f_stop_w = width
                            if stop_h > start_h and stop_w > start_w:
                                res_hand = res_hand[start_h:stop_h, start_w:stop_w, :]
                                try:
                                    bg = frame[f_start_h:f_stop_h, f_start_w:f_stop_w]
                                    blended = cv2.addWeighted(bg, alpha, res_hand, 1, 0)
                                    frame[f_start_h:f_stop_h, f_start_w:f_stop_w] = blended
                                except Exception:
                                    pass
            globals()['smoothed_hand_centers'] = smoothed_hand_centers

            # Gesture Logic
            xMinL, xMaxL, yMinL, yMaxL = None, None, None, None
            xMinR, xMaxR, yMinR, yMaxR = None, None, None, None
            
            if len(hand_centers) == 2:
                h1 = hand_centers[0]
                h2 = hand_centers[1]
                if h1[0] < h2[0]:
                    xMinR, xMaxR, yMinR, yMaxR, lmR = h1
                    xMinL, xMaxL, yMinL, yMaxL, lmL = h2
                else:
                    xMinL, xMaxL, yMinL, yMaxL, lmL = h1
                    xMinR, xMaxR, yMinR, yMaxR, lmR = h2
            
            do_predict_this_frame = (_frame_count % PRED_EVERY_N_FRAMES == 0)
            
            if xMinL and xMinR and do_predict_this_frame:
                try:
                    xMin = min(xMinL, xMinR)
                    xMax = max(xMaxL, xMaxR)
                    yMin = min(yMinL, yMinR)
                    yMax = max(yMaxL, yMaxR)
                    
                    rh = np.array([[p.x, p.y, p.z] for p in lmR.landmark]).flatten()
                    for i in np.arange(0, 63, 3):
                        rh[i] = (rh[i] - xMin) / (xMax - xMin)
                    for i in np.arange(1, 63, 3):
                        rh[i] = (rh[i] - yMin) / (yMax - yMin)
                        
                    lh = np.array([[p.x, p.y, p.z] for p in lmL.landmark]).flatten()
                    for i in np.arange(0, 63, 3):
                        lh[i] = (lh[i] - xMin) / (xMax - xMin)
                    for i in np.arange(1, 63, 3):
                        lh[i] = (lh[i] - yMin) / (yMax - yMin)
                        
                    feats_arr = np.array([np.concatenate((rh, lh))])
                    inference.queue_features(feats_arr)
                except Exception:
                    pass
            
            if len(hand_centers) > 0 and len(hand_centers) != 2:
                 if not args.gesture_mode:
                     SHIELDS = True

            prediction, pred_prob = inference.get_latest_prediction()

            # Auto Switch Logic
            if AUTO_SWITCH_ENABLED and SHIELDS:
                if shield_active_start_time is None:
                    shield_active_start_time = time.time()
                
                elapsed_active = time.time() - shield_active_start_time
                seconds_left = max(0, EFFECT_CYCLE_DURATION - elapsed_active)
                
                if seconds_left <= 0:
                    current_idx = effects_files.index(shield_loaded_name) if shield_loaded_name in effects_files else 0
                    next_idx = (current_idx + 1) % len(effects_files)
                    next_effect = effects_files[next_idx]
                    
                    shield_loaded_name = next_effect
                    effects.start_loader_for_filename(next_effect, effects_folder, args.buffer_seconds)
                    shield_active_start_time = time.time()
                    last_effect_switch_time = time.time()
                    print(f"🔄 Auto-switching to: {next_effect}")
                
                if COUNTDOWN_DURATION > 0 and seconds_left <= COUNTDOWN_DURATION:
                    frame = ui.draw_countdown_headline(frame, seconds_left, EFFECT_CYCLE_DURATION)
            else:
                shield_active_start_time = None

            # UI Overlays
            if tutorial_enabled and not SHIELDS:
                frame = ui.draw_instruction_text(frame, "RAISE BOTH HANDS")
                frame = ui.draw_hand_guides(frame, xMinL is not None, xMinR is not None)
                frame = ui.draw_detection_status(frame, xMinL is not None, xMinR is not None)
                
                # Distance Feedback
                distance_status = "OK"
                if xMinL is not None or xMinR is not None:
                    # Calculate max hand height relative to screen height
                    max_h_ratio = 0
                    if xMinL is not None:
                        max_h_ratio = max(max_h_ratio, yMaxL - yMinL)
                    if xMinR is not None:
                        max_h_ratio = max(max_h_ratio, yMaxR - yMinR)
                    
                    if max_h_ratio < config.MIN_HAND_SIZE:
                        distance_status = "TOO_FAR"
                    elif max_h_ratio > config.MAX_HAND_SIZE:
                        distance_status = "TOO_CLOSE"
                
                frame = ui.draw_distance_feedback(frame, distance_status)
            
            if show_help_overlay:
                frame = ui.draw_help_overlay(frame)

            if SHIELDS and shield_loaded_name:
                current_time = time.time()
                if last_effect_switch_time is None:
                    last_effect_switch_time = current_time
                
                time_since_switch = current_time - last_effect_switch_time
                
                if time_since_switch < EFFECT_NAME_FADE_IN_DURATION:
                    effect_name_opacity = time_since_switch / EFFECT_NAME_FADE_IN_DURATION
                elif time_since_switch < (EFFECT_NAME_FADE_IN_DURATION + EFFECT_NAME_DISPLAY_DURATION):
                    effect_name_opacity = 1.0
                elif time_since_switch < (EFFECT_NAME_FADE_IN_DURATION + EFFECT_NAME_DISPLAY_DURATION + EFFECT_NAME_FADE_OUT_DURATION):
                    fade_out_progress = (time_since_switch - EFFECT_NAME_FADE_IN_DURATION - EFFECT_NAME_DISPLAY_DURATION) / EFFECT_NAME_FADE_OUT_DURATION
                    effect_name_opacity = 1.0 - fade_out_progress
                else:
                    effect_name_opacity = 0.0
                
                if effect_name_opacity > 0:
                    frame = ui.draw_effect_name(frame, shield_loaded_name, effect_name_opacity)
            
            if xMinL or xMinR:
                last_activity_time = time.time()
            elif last_activity_time and (time.time() - last_activity_time > AUTO_RESET_SECONDS):
                SHIELDS = False
                KEY_1 = False
                KEY_3 = False
                last_activity_time = None
                session_state = "WELCOME"

            if show_window:
                cv2.imshow('Dr. Strange shields', frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key == ord('a') or key == ord('A'):
                        auto_brightness_enabled = not auto_brightness_enabled
                        if auto_brightness_enabled:
                            last_auto_brightness_time = 0
                    elif key == ord('h') or key == ord('H'):
                        show_help_overlay = not show_help_overlay
                    elif key == ord('g'):
                        SHIELDS = not SHIELDS
                    elif key == ord('f'):
                        cv2.setWindowProperty("Mystic Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    elif key == ord('r'):
                        cv2.setWindowProperty("Mystic Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    elif key == ord('m'):
                        try:
                            cv2.resizeWindow("Mystic Vision", 200, 120)
                        except Exception:
                            pass
                    elif key == ord('M'):
                        cv2.setWindowProperty("Mystic Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    elif key == ord('x') or key == ord('q'):
                        break

            if use_virtual_cam and cam:
                cam.send(frame)
                cam.sleep_until_next_frame()
            
            last_frame_time = time.time()
            sleep_time = frame_time - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n🛑 Interruption received - closing...")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
    finally:
        state.stop_threads = True
        try:
            inf_thread.join(timeout=1.0)
        except Exception:
            pass
        for loader in list(effects.shield_loaders.values()):
            try:
                loader.stop()
            except Exception:
                pass
        print("\n\n" + "="*60)
        print("\n🧹 Final cleanup...\n")
        camera_thread.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if cam:
            try:
                cam.close()
            except Exception:
                pass
        print("\n🏁 Application terminated\n")
