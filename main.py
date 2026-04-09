import os
import cv2
import json
import time
import pickle
import threading
import mediapipe as mp
import numpy as np
import requests

from flask import Flask, send_from_directory, Response, jsonify, request, stream_with_context, Blueprint
from flask_cors import CORS

# =========================================================
# 🔹 App Setup
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(base_dir, "assets"), static_url_path="/assets")
CORS(app)

# =========================================================
# 🔹 SIGN LANGUAGE MODULE (port 5000)
# =========================================================
sign_bp = Blueprint("sign", __name__)

with open(os.path.join(base_dir, "sign_language_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(base_dir, "letter_mapping.pkl"), "rb") as f:
    letter_mapping = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = None
running = False
lock = threading.Lock()

state = {"current_letter": "", "word": "", "sentence": ""}
_prev_letter = ""
_same_count = 0
STABLE_FRAMES = 6

def _reset_text_state():
    global state, _prev_letter, _same_count
    with lock:
        state = {"current_letter": "", "word": "", "sentence": ""}
    _prev_letter = ""
    _same_count = 0

def _append_letter(letter: str):
    global state
    if not letter:
        return
    with lock:
        L = letter.strip()
        if L.lower() in ["space", "_", "<space>"]:
            state["word"] += " " if not state["word"].endswith(" ") else ""
            return
        if L.lower() in ["del", "delete", "<del>"]:
            state["word"] = state["word"][:-1]
            return
        if len(L) == 1 and L.isalpha():
            state["word"] += L
        if state["word"].endswith("."):
            state["sentence"] = (state["sentence"] + " " + state["word"]).strip()
            state["word"] = ""

def _predict_from_landmarks(lms_flat):
    pred = model.predict([lms_flat])[0]
    return str(letter_mapping[pred]) if pred in letter_mapping else str(pred)

def gen_frames():
    global cap, running, _prev_letter, _same_count
    while running and cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        predicted_letter = ""
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(xs), min(ys)
            lms_flat = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
            lms_flat = [c for xy in lms_flat for c in xy]
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
        with lock:
            overlay = f"Letter: {state['current_letter']}"
        cv2.putText(frame, overlay, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@sign_bp.route("/")
def serve_sign_html():
    return send_from_directory(base_dir, "sign.html")

@sign_bp.route("/start", methods=["POST"])
def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        _reset_text_state()
    return jsonify({"ok": True, "msg": "Camera started"})

@sign_bp.route("/stop", methods=["POST"])
def stop_camera():
    global cap, running
    running = False
    if cap: cap.release()
    cap = None
    return jsonify({"ok": True, "msg": "Camera stopped"})

@sign_bp.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@sign_bp.route("/stream")
def stream():
    @stream_with_context
    def event_gen():
        while True:
            with lock: payload = json.dumps(state)
            yield f"data: {payload}\n\n"
            time.sleep(0.1)
    return Response(event_gen(), mimetype="text/event-stream")

@sign_bp.route("/clear", methods=["POST"])
def clear_state():
    _reset_text_state()
    return jsonify({"ok": True})

# =========================================================
# 🔹 AI VOICEBOT MODULE (port 5001)
# =========================================================
voice_bp = Blueprint("voice", __name__)

API_KEY = "YOUR_GEMINI_KEY"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def chatbot_response(query):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": query}]}]}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "I couldn't understand Gemini's reply."
    return f"Error {response.status_code}: {response.text}"

@voice_bp.route("/")
def serve_voice_html():
    return send_from_directory(base_dir, "voicebot.html")

@voice_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"reply": "No message received."}), 400
    return jsonify({"reply": chatbot_response(user_message)})

# =========================================================
# 🔹 Register Blueprints & Run Servers
# =========================================================
app.register_blueprint(sign_bp, url_prefix="/signapp")
app.register_blueprint(voice_bp, url_prefix="/voicebot")

if __name__ == "__main__":
    from threading import Thread

    def run_sign():
        app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)

    def run_voice():
        app.run(host="0.0.0.0", port=5001, debug=True, threaded=True, use_reloader=False)

    Thread(target=run_sign).start()
    Thread(target=run_voice).start()
    