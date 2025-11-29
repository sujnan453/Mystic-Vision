import threading

# Shared state for initialization status
init_status = {"step": "Starting...", "progress": 0}
init_lock = threading.Lock()

# Global flag to stop all threads
stop_threads = False

def update_init_status(step, progress):
    """Update the initialization status safely across threads."""
    global init_status
    with init_lock:
        init_status = {"step": step, "progress": progress}
