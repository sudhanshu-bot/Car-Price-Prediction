from flask import Flask, jsonify, request

from model_pipeline import MODEL_PATH, predict_one, train_and_save

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object containing car specifications."}), 400
    try:
        return jsonify({"predicted_price": predict_one(payload)})
    except FileNotFoundError:
        return jsonify({"error": "Model is not trained. Run model_pipeline.py first."}), 503
    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        train_and_save()
    app.run(host="127.0.0.1", port=5000, debug=False)