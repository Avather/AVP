
# AvaTherVisionPro 🚦📱

**AvaTherVisionPro** is a Flask-based assistive AI backend app designed for visually impaired users. It uses Teachable Machine models to detect obstacles, assist in road crossings, recognize Indian currency, and provide real-time audio navigation — and is fully integrable with MIT App Inventor for mobile accessibility.

---

## 🌟 Features

1. **Obstacle Detection (100 meters)**  
   Detects animals, people, vehicles, fruits, and static objects like furniture. Works in day and night (activates flashlight in low light).

2. **Audio Navigation**  
   Provides spoken directions like "Turn Left", "Go Forward", "Obstacle Ahead", based on the user's path and obstacle detection.

3. **Road Crossing Assistance**  
   Analyzes surroundings for traffic lights, vehicles, and zebra crossings. Alerts if it’s safe or unsafe to cross.

4. **Currency Detection (Indian Rupees)**  
   Detects ₹10, ₹20, ₹50, ₹100, ₹200, ₹500 using front/back-side images with voice output.

5. **Voice Commands and Flashlight Auto-Activation** *(Coming Soon)*  
   Voice-based activation and automatic flashlight control based on ambient light levels.

---

## 📁 Project Structure

```
AvaTherVisionPro/
├── app.py                       # Flask backend server
├── requirements.txt             # Python dependencies
├── modules/
│   └── speak_helper.py          # Text-to-speech engine using pyttsx3
├── static/
│   └── models/                  # TFLite models and label files
│       ├── obstacle_detection.tflite
│       ├── obstacle_detection_labels.txt
│       ├── road_crossing_assistance.tflite
│       ├── road_crossing_assistance_labels.txt
│       ├── currency_detection.tflite
│       ├── currency_detection_labels.txt
├── flaskenv/                    # Python virtual environment
└── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

### ✅ 1. Clone & Enter Project

```bash
git clone https://github.com/<your-username>/AvaTherVisionPro.git
cd AvaTherVisionPro
```

### ✅ 2. Create and Activate Virtual Environment

```bash
python3.10 -m venv flaskenv
source flaskenv/bin/activate
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Locally

```bash
python app.py
```

Then access the Flask app at:  
`http://127.0.0.1:5050`

### API Endpoints

- `POST /predict/obstacle_detection`
- `POST /predict/road_crossing_assistance`
- `POST /predict/currency_detection`

> Make sure to POST an image file with key `"image"` in your test request.

---

## 🌐 Deployment on PythonAnywhere

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your project or clone from GitHub
3. Create a **Web App** on PythonAnywhere
4. Set `flaskenv` as your virtualenv path
5. Point WSGI to `app.py`
6. Reload Web App

Your backend will be accessible at:  
`https://your-username.pythonanywhere.com/predict/<model_name>`

---

## 🤖 MIT App Inventor Integration

- Use the **Web component** to send image files to `/predict/<model_name>`
- Use **TextToSpeech** to read the response label
- Integrate camera and flashlight control as per Android permissions

---

## 🧠 Model Format

- All models are exported from **Teachable Machine**
- Use **Quantized format (.tflite)** during export
- Input shape is `224x224` and normalized to [0.0, 1.0]
- Labels should be saved in corresponding `<model_name>_labels.txt`

---

## 📢 Audio System

Speech responses use `pyttsx3`, configured to avoid the “run loop already started” error using thread-safe logic.  
Only essential labels like "Safe to Cross", "Do Not Cross", or "Obstacle Ahead" are spoken.

---

## 🛠 Tech Stack

- Python 3.10
- Flask 2.2.5
- TensorFlow Lite
- pyttsx3
- Pillow
- MIT App Inventor (frontend)

---

## 📝 License

Open source for academic and research use. Attribution appreciated.  
Ensure proper compliance when deploying for production or public use.

---

## 🙏 Credits

- [Google Teachable Machine](https://teachablemachine.withgoogle.com/)
- [PythonAnywhere](https://www.pythonanywhere.com/)
- [MIT App Inventor](https://appinventor.mit.edu/)
