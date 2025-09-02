#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# flask_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load your trained model
MODEL_PATH = "melanoma_cnn_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # Set model to None to handle cases where the file is missing
    model = None

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Preprocessing function (adjust target_size to your model input)
def preprocess_image(img, target_size=(224, 224)):
    """Preprocesses a PIL Image for model prediction."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize if your model was trained this way
    return img_array

# Define recommendations and clinical guidelines based on the prediction
recommendations = {
    "melanoma": {
        "text": "Based on the prediction, this lesion has characteristics of a melanoma. It is **strongly recommended** that you consult with a dermatologist or a qualified healthcare professional for a physical examination and further diagnosis. **This tool is for informational purposes only and is not a substitute for professional medical advice.**",
        "action": "Immediate consultation with a dermatologist is advised."
    },
    "benign": {
        "text": "The analysis suggests this lesion is likely benign. However, it's crucial to perform regular self-examinations and monitor for any changes in size, shape, color, or texture. If you notice any changes, please consult a healthcare professional. **This tool is for informational purposes only and is not a substitute for professional medical advice.**",
        "action": "Continue self-monitoring. Consult a doctor if any changes occur."
    }
}

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to handle image uploads and return predictions with recommendations."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the model file path."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    processed_img = preprocess_image(img)
    prediction_value = model.predict(processed_img)[0][0]
    
    # Determine the label and recommendation based on a 0.5 threshold
    if prediction_value > 0.5:
        label = "Melanoma"
        confidence = float(prediction_value)
        recommendation = recommendations["melanoma"]["text"]
        clinical_guideline = recommendations["melanoma"]["action"]
    else:
        label = "Benign"
        confidence = float(1 - prediction_value)
        recommendation = recommendations["benign"]["text"]
        clinical_guideline = recommendations["benign"]["action"]

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "recommendation": recommendation,
        "clinical_guideline": clinical_guideline
    })

# Important: run Flask in main thread and disable reloader
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

