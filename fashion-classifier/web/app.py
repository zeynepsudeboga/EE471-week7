import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

# Address of the Cog model container. Inside docker compose it is reachable by its service name.
COG_URL = os.getenv("COG_URL", "http://localhost:5050")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/classify")
def classify():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="Please choose an image first."), 400
    if not (file.mimetype or "").startswith("image/"):
        return jsonify(error="That file is not an image."), 400

    # Cog accepts files as data URIs inside the JSON body
    data_uri = f"data:{file.mimetype};base64,{base64.b64encode(file.read()).decode()}"
    try:
        r = requests.post(f"{COG_URL}/predictions", json={"input": {"image": data_uri}}, timeout=120)
    except requests.RequestException:
        return jsonify(error="The model container is not reachable."), 502

    if r.status_code in (409, 503):
        return jsonify(error="The model is still starting up. Try again in a few seconds."), 503
    body = r.json()
    if r.status_code != 200 or body.get("status") != "succeeded":
        return jsonify(error=body.get("error") or "Prediction failed."), 502
    return jsonify(body["output"])


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 10 MB."), 413
