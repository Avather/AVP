import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, session
from flask_cors import CORS

import tflite_runtime.interpreter as tflite
from scipy.special import softmax

app = Flask(__name__)
CORS(app)
app.secret_key = "ava_vision_secret"

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "static", "models")
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

        interpreter = tflite.Interpreter(model_path=model_path)
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

@app.route("/set_mode", methods=["POST"])
def set_mode():
    data = request.get_json()
    mode = data.get("mode")
    if mode not in ["obstacle_detection", "currency_detection", "road_crossing_assistance"]:
        return jsonify({"error": "Invalid mode"}), 400
    session["active_mode"] = mode
    return jsonify({"message": f"Mode set to {mode}"})


@app.route("/voice_command", methods=["POST"])
def voice_command():
    command = request.json.get("command", "").lower()

    if "stop" in command:
        session.pop("active_mode", None)
        return jsonify({"message": "Current mode stopped"})

    elif "currency detection" in command:
        session["active_mode"] = "currency_detection"
    elif "obstacle detection" in command:
        session["active_mode"] = "obstacle_detection"
    elif "road crossing assistance" in command or "road" in command or "crossing" in command:
        session["active_mode"] = "road_crossing_assistance"
    else:
        return jsonify({"error": "Unrecognized command"}), 400

    return jsonify({"message": f"Mode set via voice to {session['active_mode']}"})

@app.route("/predict", methods=["POST"])
def predict():
    model_name = session.get("active_mode")
    if not model_name:
        return jsonify({"error": "No mode set. Please call /set_mode or /voice_command first."}), 400

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

        if input_type == np.uint8:
            predictions = output
        else:
            predictions = softmax(output).tolist()

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"

        confidence_per_class = [
            {"label": lbl, "confidence": round(float(conf), 3)}
            for lbl, conf in zip(labels, predictions)
        ]

        voice_text = ""
        if model_name == "road_crossing_assistance":
            if "do_not_cross" in label.lower() or "red_light" in label.lower():
                voice_text = "Do Not Cross"
            elif "safe_to_cross" in label.lower() or "green_light" in label.lower():
                voice_text = "Safe to Cross"
            elif "vehicle_detected" in label.lower():
                voice_text = "Vehicle ahead, please stop"
            elif "clear_road" in label.lower():
                voice_text = "Road is clear, proceed carefully"
        elif model_name == "currency_detection":
            voice_text = f"{label} detected"
        elif model_name == "obstacle_detection":
            voice_text = f"Obstacle ahead: {label}"

        # Return only voice_text if requested (audio_only=true)
        if request.args.get("audio_only") == "true":
            return jsonify({"voice_text": voice_text})

        return jsonify({
            "predicted_class": predicted_index,
            "label": label,
            "confidence": round(confidence, 3),
            "confidence_per_class": confidence_per_class,
            "voice_text": voice_text
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route("/")
def index():
    return (
        "<h2>✅ AvaTherVisionPro API is running</h2>"
        "<p>Use <code>POST /set_mode</code> or <code>/voice_command</code>.</p>"
        "<p>Then <code>POST /predict</code> with an image file.</p>"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
