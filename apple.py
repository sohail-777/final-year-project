from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import os
import json
import matplotlib.pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load sentiment model
sentiment_analyzer = pipeline(
    "text-classification", model="tabularisai/multilingual-sentiment-analysis")

# Data file
DATA_FILE = "analyzed.json"
GRAPH_FOLDER = "static/graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Load existing data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        products = json.load(f)
else:
    products = []


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    name = data.get("name")
    pid = data.get("id")
    image = data.get("image")
    reviews = data.get("reviews", [])

    if not name or not pid or not reviews:
        return jsonify({"error": "Missing required fields"}), 400

    sentiments = sentiment_analyzer(reviews)
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in sentiments:
        label = s['label'].lower()
        sentiment_counts[label] += 1

    overall = max(sentiment_counts, key=sentiment_counts.get)

    product = {
        "name": name,
        "id": pid,
        "image": image,
        "sentiment": overall,
        "counts": sentiment_counts
    }

    # Remove old one with same ID (to update)
    global products
    products = [p for p in products if p["id"] != pid]
    products.append(product)

    with open(DATA_FILE, "w") as f:
        json.dump(products, f)

    # Plot bar chart
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    plt.figure()
    plt.bar(labels, values, color=["green", "gray", "red"])
    plt.title(f"Sentiment for {name}")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    graph_path = os.path.join(GRAPH_FOLDER, f"{pid}.png")
    plt.savefig(graph_path)
    plt.close()

    return jsonify({"sentiment": overall})


@app.route("/get-products")
def get_products():
    return jsonify(products)


@app.route("/get-graph/<pid>")
def get_graph(pid):
    path = os.path.join(GRAPH_FOLDER, f"{pid}.png")
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Graph not found", 404


@app.route("/delete/<pid>", methods=["DELETE"])
def delete_product(pid):
    global products
    products = [p for p in products if p["id"] != pid]

    with open(DATA_FILE, "w") as f:
        json.dump(products, f)

    # Delete graph
    graph_path = os.path.join(GRAPH_FOLDER, f"{pid}.png")
    if os.path.exists(graph_path):
        os.remove(graph_path)

    return jsonify({"message": "Deleted"}), 200


if __name__ == "__main__":
    app.run(debug=True)
