"""
app.py  –  Flask API for isometric room rendering
==================================================
Endpoints:

  POST /isometric
    Body (JSON):  { "url": "https://..." }
    Returns:      PNG image bytes  (Content-Type: image/png)

  GET /health
    Returns:      { "status": "ok" }
"""

import os
import requests
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
import io

from process import generate_isometric

app = Flask(__name__)

MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB cap


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/isometric", methods=["POST"])
def isometric():
    # ── Parse request ─────────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Provide JSON body: {\"url\": \"https://...\"}"}), 400

    url = data["url"].strip()

    # ── Download image ────────────────────────────────────────────────────────
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > MAX_IMAGE_BYTES:
            return jsonify({"error": "Image too large (max 20 MB)"}), 413

        raw = b""
        for chunk in resp.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) > MAX_IMAGE_BYTES:
                return jsonify({"error": "Image too large (max 20 MB)"}), 413
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download image: {e}"}), 400

    # ── Decode with OpenCV ────────────────────────────────────────────────────
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image. "
                                 "Send a valid JPEG/PNG URL."}), 422

    # ── Generate isometric render ─────────────────────────────────────────────
    try:
        png_bytes = generate_isometric(img_bgr)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    # ── Return PNG ────────────────────────────────────────────────────────────
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="isometric.png"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
