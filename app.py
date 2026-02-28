from flask import Flask, render_template, request, jsonify
import os
import random
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    # Simulated AI detection logic
    variance = np.var(img_array)

    # Convert variance into probability score (10–95%)
    probability = min(max(int((variance % 100)), 10), 95)

    # Risk classification
    if probability > 70:
        risk = "HIGH"
    elif probability > 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return probability, risk


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    probability, risk = analyze_image(file_path)

    return jsonify({
        "probability": probability,
        "risk": risk
    })


if __name__ == "__main__":
    app.run(debug=True)
