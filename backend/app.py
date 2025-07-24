from flask import Flask, render_template, request
from detector import detect_deepfake
import os
from PIL import Image

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = None
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join("static", "uploaded.jpg")
        file.save(filepath)
        prediction,confidence,inference_time = detect_deepfake(filepath)
    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run()

