# app.py
import os
import joblib
from flask import Flask, request, jsonify, render_template


MODEL_PATH = os.path.join("models", "spam_model.pkl")
VEC_PATH = os.path.join("models", "vectorizer.pkl")


app = Flask(__name__)


# Load artifacts once at startup
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)




@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Message is required."}), 400


        X = vectorizer.transform([message])
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0][1]) # probability of spam


        return jsonify({
        "ok": True,
        "label": "Spam" if pred == 1 else "Ham",
        "spam_probability": round(proba, 4)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)