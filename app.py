from constants import CLASS_NAMES, DISEASE_INFO
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import traceback

# If you already implemented Grad-CAM, keep this import safe
try:
    from gradcam_utils import generate_all_gradcams
    GRADCAM_AVAILABLE = True
except:
    GRADCAM_AVAILABLE = False
    print("GradCAM module not found or failed to import")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# Models
# ----------------------------
MODEL_PATHS = {
    "DenseNet121": "backend-fyp/DenseNet121_finetune.h5",
    "ResNet50": "backend-fyp/ResNet50_finetune.h5",
    "EfficientNetB0": "backend-fyp/EffNetB0_finetuned_final.h5"
}

models_dict = {}

for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models_dict[name] = tf.keras.models.load_model(path)
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Failed loading {name}: {e}")
    else:
        print(f"Model not found: {path}")

# ----------------------------
# Class names
# ----------------------------
CLASS_NAMES = [
    "Angular_Leafspot",
    "Anthracnose_Fruit_Rot",
    "Blossom_Blight",
    "Gray_Mold",
    "Leaf_Blight",
    "Leaf_Spot",
    "Powdery_Mildew_Fruit",
    "Powdery_Mildew_Leaf"
]

# ----------------------------
# Image preprocessing
# ----------------------------
IMG_SIZE = (224, 224)

def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    return img

# ----------------------------
# Prediction route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        img = preprocess_image(file)

        if not models_dict:
            return jsonify({"error": "No models loaded"}), 500

        individual = []
        all_probs = []

        for name, model in models_dict.items():
            preds = model.predict(img, verbose=0)[0]

            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx] * 100)

            individual.append({
                "model": name,
                "disease": CLASS_NAMES[class_idx],
                "confidence": round(confidence, 2)
            })

            all_probs.append(preds)

        avg_probs = np.mean(all_probs, axis=0)

        ensemble_idx = int(np.argmax(avg_probs))
        ensemble_conf = float(avg_probs[ensemble_idx] * 100)

        return jsonify({
            "individual": individual,
            "ensemble": {
                "disease": CLASS_NAMES[ensemble_idx],
                "confidence": round(ensemble_conf, 2)
            }
        })

    except Exception as e:
        print(" PREDICT ERROR:")
        traceback.print_exc()

        return jsonify({"error": str(e)}), 500


# ----------------------------
# Grad-CAM route — runs on ALL 3 models, returns one heatmap per model
# ----------------------------
@app.route("/gradcam", methods=["POST"])
def gradcam():
    try:
        if not models_dict:
            return jsonify({"error": "Models not loaded"}), 500

        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        if not GRADCAM_AVAILABLE:
            return jsonify({"error": "Grad-CAM not available"}), 500

        img = preprocess_image(file)

        results = generate_all_gradcams(img, models_dict, CLASS_NAMES)

        # Also compute ensemble prediction
        all_probs = []
        for name, model in models_dict.items():
            preds = model.predict(img, verbose=0)[0]
            all_probs.append(preds)

        avg_probs = np.mean(all_probs, axis=0)
        ensemble_idx = int(np.argmax(avg_probs))
        ensemble_conf = round(float(avg_probs[ensemble_idx] * 100), 2)

        return jsonify({
            "models": results,
            "ensemble": {
                "disease": CLASS_NAMES[ensemble_idx],
                "confidence": ensemble_conf
            }
        })

    except Exception as e:
        print(" GRADCAM ROUTE ERROR:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)