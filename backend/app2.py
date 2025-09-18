from flask import Flask, request, jsonify
from datetime import datetime
import os, csv, time
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
RESULTS_CSV = "results.csv"

# Load models once at startup
cow_model = YOLO("../models/cow_detect_prototype.pt")
buffalo_model = YOLO("../models/buffalo_delect_prototype.pt")

def save_result_row(row):
    header = ["timestamp", "image", "animal", "breed", "confidence"]
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

@app.route("/")
def home():
    return "Cattle Classification Backend — Local Demo"

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    if "animal" not in request.form:
        return jsonify({"error": "No animal type provided"}), 400

    animal_type = request.form["animal"].lower()
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(UPLOAD_DIR, f"{ts}_{file.filename.replace(' ', '_')}")
    file.save(save_path)

    # Run the correct model
    if animal_type == "cow":
        results = cow_model(save_path)
    elif animal_type == "buffalo":
        results = buffalo_model(save_path)
    else:
        return jsonify({"error": "Invalid animal type"}), 400

    # Extract breed info
    top = results[0]
    if len(top.boxes) == 0:
        return jsonify({"error": "No animal detected"}), 200

    class_id = int(top.boxes.cls[0].item())
    breed = top.names[class_id]
    conf = float(top.boxes.conf[0].item())

    result = {
        "animal": animal_type,
        "breed": breed,
        "confidence": round(conf, 3),
        "image_saved_to": save_path
    }

    save_result_row({
        "timestamp": int(time.time()),
        "image": save_path,
        "animal": animal_type,
        "breed": breed,
        "confidence": round(conf, 3)
    })

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
