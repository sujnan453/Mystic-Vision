import threading
import queue
import time
import numpy as np
import pickle
from state import update_init_status, stop_threads
from config import config

features_q = queue.Queue(maxsize=1)
pred_lock = threading.Lock()
latest_prediction = None
latest_pred_prob = 0.0

def load_model(model_path=None):
    """Load the SVM model from disk"""
    if model_path is None:
        model_path = config.MODEL_PATH
        
    update_init_status("Loading ML model...", 30)
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def inference_worker(model):
    global latest_prediction, latest_pred_prob
    
    update_init_status("Starting inference engine...", 93)
    
    while not stop_threads:
        try:
            feats = features_q.get(timeout=0.1)
        except queue.Empty:
            continue
        
        if model:
            try:
                pred = model.predict(feats)[0]
                prob = float(np.max(model.predict_proba(feats)))
            except Exception:
                pred = None
                prob = 0.0
        else:
            pred = None
            prob = 0.0
            
        with pred_lock:
            latest_prediction = pred
            latest_pred_prob = prob
        time.sleep(0.001)

def start_inference_thread(model):
    inf_thread = threading.Thread(target=inference_worker, args=(model,), daemon=True)
    inf_thread.start()
    return inf_thread

def get_latest_prediction():
    with pred_lock:
        return latest_prediction, latest_pred_prob

def queue_features(feats_arr):
    try:
        features_q.put_nowait(feats_arr)
    except queue.Full:
        try:
            _ = features_q.get_nowait()
        except queue.Empty:
            pass
        try:
            features_q.put_nowait(feats_arr)
        except queue.Full:
            pass
