from flask import Flask, request, jsonify
import requests
import json
from flask_cors import CORS   # 👈 allow frontend to call backend

# Gemini API Setup
API_KEY = "AIzaSyDoErOe37jYWUKMdtfGm353BEEYgts6t9M"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

app = Flask(__name__)
CORS(app)   # 👈 allow cross-origin (your HTML is served separately)

def chatbot_response(query):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": query}]}]}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "I couldn't understand Gemini's reply."
    else:
        return f"Error {response.status_code}: {response.text}"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"reply": "No message received."}), 400
    reply = chatbot_response(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
