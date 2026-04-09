import os
import cv2
import json
import time
import pickle
import threading
import mediapipe as mp
import numpy as np
from flask import Flask, send_from_directory, Response, jsonify, request, stream_with_context
from spellchecker import SpellChecker   # 🔹 NEW

app = Flask(__name__)

# -------------------------------
# Load model + mappings
# -------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, "sign_language_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(base_dir, "letter_mapping.pkl"), "rb") as f:
    letter_mapping = pickle.load(f)

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Camera + prediction state
# -------------------------------
cap = None
running = False
lock = threading.Lock()

state = {
    "current_letter": "",
    "word": "",
    "sentence": ""
}

_prev_letter = ""
_same_count = 0
STABLE_FRAMES = 6             # stable detection before appending a letter
NO_DETECTION_FRAMES = 30      # finalize word after N missing frames
_no_det_count = 0             # internal counter

# 🔹 Spell checker instance
spell = SpellChecker()


def _reset_text_state():
    global state, _prev_letter, _same_count, _no_det_count
    with lock:
        state["current_letter"] = ""
        state["word"] = ""
        state["sentence"] = ""
    _prev_letter = ""
    _same_count = 0
    _no_det_count = 0


def _append_letter(letter: str):
    global state
    if not letter:
        return
    with lock:
        L = letter.strip()
        if len(L) == 1 and L.isalpha():
            state["word"] += L


def _finalize_word():
    """Spell-correct the current word, move to sentence, and reset."""
    global state
    with lock:
        raw_word = state["word"].strip()
        if raw_word:
            corrected = spell.correction(raw_word) or raw_word
            state["sentence"] = (state["sentence"] + " " + corrected).strip()
            state["word"] = ""


def _predict_from_landmarks(lms_flat):
    pred = model.predict([lms_flat])[0]
    if pred in letter_mapping:
        return str(letter_mapping[pred])
    return str(pred)


def gen_frames():
    """
    MJPEG generator for /video_feed.
    Updates text state with debouncing and finalization.
    """
    global cap, running, _prev_letter, _same_count, _no_det_count

    while running and cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_letter = ""

        if results.multi_hand_landmarks:
            _no_det_count = 0
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(xs), min(ys)

            lms_flat = []
            for lm in hand_landmarks.landmark:
                lms_flat.extend([lm.x - min_x, lm.y - min_y])

            predicted_letter = _predict_from_landmarks(lms_flat)

            if predicted_letter == _prev_letter:
                _same_count += 1
            else:
                _prev_letter = predicted_letter
                _same_count = 1

            with lock:
                state["current_letter"] = predicted_letter

            if _same_count >= STABLE_FRAMES:
                _append_letter(predicted_letter)
                _same_count = 0

        else:
            # No detection → increment counter
            _no_det_count += 1
            if _no_det_count >= NO_DETECTION_FRAMES:
                _finalize_word()
                _no_det_count = 0

        with lock:
            overlay = f"Letter: {state['current_letter']}"
        cv2.putText(frame, overlay, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def root():
    return send_from_directory(base_dir, "sign.html")


@app.route("/sign")
def serve_sign():
    return send_from_directory(base_dir, "sign.html")


@app.route("/start", methods=["POST", "GET"])
def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        _reset_text_state()
        return jsonify({"ok": True, "msg": "Camera started"})
    return jsonify({"ok": True, "msg": "Camera already running"})


@app.route("/stop", methods=["POST", "GET"])
def stop_camera():
    global cap, running
    running = False
    if cap:
        try:
            cap.release()
        except Exception:
            pass
        cap = None
    return jsonify({"ok": True, "msg": "Camera stopped"})


@app.route("/video_feed")
def video_feed():
    global running
    if not running:
        start_camera()
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream")
def stream():
    @stream_with_context
    def event_gen():
        while True:
            with lock:
                payload = json.dumps(state)
            yield f"data: {payload}\n\n"
            time.sleep(0.1)

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(event_gen(), headers=headers)


@app.route("/clear", methods=["POST"])
def clear_state():
    _reset_text_state()
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"reply": "Hi! Ask me anything about the prediction or app."})
    if "hello" in user_msg.lower():
        return jsonify({"reply": "Hello! How can I help you today?"})
    return jsonify({"reply": f"You said: {user_msg}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)