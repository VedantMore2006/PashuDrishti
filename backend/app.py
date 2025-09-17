# app.py
from flask import Flask, request, jsonify
from datetime import datetime
import os
import csv
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Make a folder to save uploaded images (demo)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULTS_CSV = "results.csv"


def save_result_row(row):
    header = ["timestamp", "image", "animal", "confidence", "score"]
    fname = RESULTS_CSV
    exists = os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


@app.route("/")
def home():
    return "Cattle Classification Backend — Local Demo"


# Simple classify endpoint that accepts an image file
@app.route("/classify", methods=["POST"])
def classify():
    # Check file present
    if "file" not in request.files:
        return jsonify({"error": "No file key in request. Use 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image locally (for demo)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = file.filename.replace(" ", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{ts}_{safe_filename}")
    file.save(save_path)

    # ---------- DUMMY MODEL OUTPUT ----------
    # Replace this block with Vedant's inference code later:
    # Example:
    # result = run_inference(save_path)
    # Ensure run_inference returns a dict with keys: animal, confidence, traits, score
    dummy_result = {
        "animal": "cow",
        "confidence": 0.92,
        "traits": {
            "body_length_cm": 140,
            "height_withers_cm": 125,
            "chest_width_cm": 60,
        },
        "score": 78,
        "image_saved_to": save_path,
    }

    # Persist result to CSV for /history
    try:
        save_result_row(
            {
                "timestamp": int(time.time()),
                "image": save_path,
                "animal": dummy_result.get("animal", ""),
                "confidence": dummy_result.get("confidence", 0),
                "score": dummy_result.get("score", 0),
            }
        )
    except Exception as e:
        # don't crash the API if saving fails; just include warning in response
        dummy_result["save_error"] = str(e)

    return jsonify(dummy_result), 200


@app.route("/history", methods=["GET"])
def history():
    """Return last N rows from results.csv as JSON."""
    fname = RESULTS_CSV
    if not os.path.exists(fname):
        return jsonify([])

    rows = []
    try:
        with open(fname, newline="") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    except Exception as e:
        return jsonify({"error": "Failed reading results.csv", "details": str(e)}), 500

    # return up to last 50 results (most recent last in CSV)
    return jsonify(rows[-50:])


if __name__ == "__main__":
    # debug=True gives helpful errors if something breaks
    app.run(host="127.0.0.1", port=5000, debug=True)
