import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from modules.speak_helper import speak  # Uses macOS 'say' for speech

import tensorflow as tf

app = Flask(__name__)

# === Configuration ===
MODEL_DIR = "static/models"
MODEL_SUFFIX = ".tflite"
LABEL_SUFFIX = "_labels.txt"
DEFAULT_INPUT_SIZE = (224, 224)

# === Cache loaded models ===
loaded_models = {}

def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def load_model(model_name):
    if model_name not in loaded_models:
        model_path = os.path.join(MODEL_DIR, f"{model_name}{MODEL_SUFFIX}")
        label_path = os.path.join(MODEL_DIR, f"{model_name}{LABEL_SUFFIX}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        labels = load_labels(label_path)
        input_type = interpreter.get_input_details()[0]['dtype']

        loaded_models[model_name] = (interpreter, labels, input_type)

    return loaded_models[model_name]

def preprocess_image(image_bytes, input_size, input_type):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(input_size)
    img_array = np.array(img)

    if input_type == np.uint8:
        return np.expand_dims(img_array, axis=0).astype(np.uint8)
    else:
        return np.expand_dims(img_array / 255.0, axis=0).astype(np.float32)

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = request.files["image"].read()

    try:
        interpreter, labels, input_type = load_model(model_name)
    except Exception as e:
        return jsonify({"error": f"Model load failed: {str(e)}"}), 500

    input_size = (224, 224)
    input_data = preprocess_image(image_bytes, input_size, input_type)

    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Apply softmax if required
        if input_type == np.uint8:
            predictions = output
        else:
            predictions = tf.nn.softmax(output).numpy()

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"

        print(f"\n📌 Prediction log for: {model_name}")
        print(f"🔢 Raw model output: {predictions}")
        print(f"✅ Predicted index: {predicted_index}")
        print(f"🏷️ Label: {label}")
        print(f"📈 Confidence: {round(confidence, 3)}")

        # Detailed confidence breakdown
        confidence_per_class = [
            {"label": lbl, "confidence": round(float(conf), 3)}
            for lbl, conf in zip(labels, predictions)
        ]

        # === Voice Feedback Logic ===
        if model_name in ["road_crossing_assistance", "enhanced_road_crossing", "vehicle_assistance"]:
            if "do_not_cross" in label.lower() or "red_light" in label.lower():
                speak("Do Not Cross")
            elif "safe_to_cross" in label.lower() or "green_light" in label.lower():
                speak("Safe to Cross")
            elif "vehicle_detected" in label.lower():
                speak("Vehicle ahead, please stop")
            elif "clear_road" in label.lower():
                speak("Road is clear, proceed carefully")
        elif model_name == "currency_detection":
            speak(f"{label} detected")
        elif model_name == "obstacle_detection":
            speak(f"Obstacle ahead: {label}")

        return jsonify({
            "predicted_class": predicted_index,
            "label": label,
            "confidence": round(confidence, 3),
            "confidence_per_class": confidence_per_class
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

