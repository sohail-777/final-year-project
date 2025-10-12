from flask import Flask, request, jsonify, send_from_directory, abort
# local module (centralized model)
from sentiment_model import analyze_sentiment
import os
import json
import matplotlib.pyplot as plt
from flask_cors import CORS
from threading import Lock
from tempfile import NamedTemporaryFile

app = Flask(__name__)
CORS(app)

# Data file (store inside data/ for clarity)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "analyzed.json")

GRAPH_FOLDER = os.path.join("static", "graphs")
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# In-memory products list and lock for safe concurrent writes
products = []
_products_lock = Lock()

# Load existing data (if any) at startup
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            products = json.load(f)
    except Exception:
        products = []


def _atomic_write_json(path, data):
    """Write JSON atomically to avoid partial writes (write temp file then replace)."""
    dirp = os.path.dirname(path)
    os.makedirs(dirp, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=dirp, encoding="utf-8") as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tempname = tf.name
    os.replace(tempname, path)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expected JSON body:
    {
      "name": "<product name>",
      "id": "<product id>",
      "image": "<image url optional>",
      "reviews": ["one review", "another review"]  OR "reviews": "one review\nanother review"
    }
    Returns JSON:
    {
      "overall": "POSITIVE" | "NEUTRAL" | "NEGATIVE",
      "counts": {"positive": n, "neutral": n, "negative": n}
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    name = data.get("name", "").strip()
    pid = str(data.get("id", "")).strip()
    image = data.get("image", "").strip() if data.get("image") else ""

    reviews = data.get("reviews", [])
    # Accept string (split into lines) or list
    if isinstance(reviews, str):
        reviews = [r.strip() for r in reviews.splitlines() if r.strip()]
    elif isinstance(reviews, list):
        reviews = [str(r).strip() for r in reviews if str(r).strip()]
    else:
        reviews = []

    if not name or not pid or not reviews:
        return jsonify({"error": "Missing required fields: name, id, reviews"}), 400

    # Analyze using central function
    try:
        counts, overall = analyze_sentiment(reviews)
    except Exception as e:
        return jsonify({"error": f"Sentiment analysis failed: {e}"}), 500

    # Normalize counts keys to lowercase for consistent API
    normalized_counts = {
        "positive": int(counts.get("POSITIVE", counts.get("positive", 0))),
        "neutral": int(counts.get("NEUTRAL", counts.get("neutral", 0))),
        "negative": int(counts.get("NEGATIVE", counts.get("negative", 0)))
    }

    product = {
        "name": name,
        "id": pid,
        "image": image,
        "sentiment": overall,
        "counts": normalized_counts
    }

    # Persist: update in-memory list and write atomically to disk
    with _products_lock:
        # remove any existing item with same id
        products[:] = [p for p in products if str(p.get("id")) != pid]
        products.append(product)
        try:
            _atomic_write_json(DATA_FILE, products)
        except Exception as e:
            return jsonify({"error": f"Failed to save data: {e}"}), 500

    # Create and save bar chart (overwrite existing)
    try:
        labels = ["positive", "neutral", "negative"]
        values = [normalized_counts.get(l, 0) for l in labels]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.title(f"Sentiment for {name}")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        graph_path = os.path.join(GRAPH_FOLDER, f"{pid}.png")
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
    except Exception:
        # Do not fail the API if plotting fails; just continue
        pass

    return jsonify({"overall": overall, "counts": normalized_counts}), 200


@app.route("/get-products", methods=["GET"])
def get_products():
    # Return the in-memory list (safe snapshot)
    with _products_lock:
        return jsonify(products), 200


@app.route("/get-graph/<pid>", methods=["GET"])
def get_graph(pid):
    graph_path = os.path.join(GRAPH_FOLDER, f"{pid}.png")
    if os.path.exists(graph_path):
        # send_from_directory for safer serving
        return send_from_directory(GRAPH_FOLDER, f"{pid}.png")
    return abort(404, description="Graph not found")


@app.route("/delete/<pid>", methods=["DELETE"])
def delete_product(pid):
    with _products_lock:
        global products
        products = [p for p in products if str(p.get("id")) != str(pid)]
        try:
            _atomic_write_json(DATA_FILE, products)
        except Exception as e:
            return jsonify({"error": f"Failed to save data after deletion: {e}"}), 500

    # Delete graph file if exists
    graph_path = os.path.join(GRAPH_FOLDER, f"{pid}.png")
    if os.path.exists(graph_path):
        try:
            os.remove(graph_path)
        except Exception:
            pass

    return jsonify({"message": "Deleted"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
