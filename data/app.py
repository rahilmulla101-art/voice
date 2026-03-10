from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime

# -----------------------------------------
# CONFIG
# -----------------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "emotion_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------------------
# LOAD MODEL & ENCODER
# -----------------------------------------
print("🔄 Loading model and encoder...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

print("✅ Model and encoder loaded!")


# -----------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------
def extract_features(file_path, max_duration=3):
    y, sr = librosa.load(file_path, sr=None)

    max_len = sr * max_duration

    # Pad or trim audio
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")

    # Extract MFCC, Chroma, Tonnetz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Combine features
    features = np.concatenate([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(tonnetz.T, axis=0)
    ])

    return features


# -----------------------------------------
# PREDICT
# -----------------------------------------
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    idx = np.argmax(prediction)
    emotion = le.inverse_transform([idx])[0]

    return emotion, prediction[0]


# -----------------------------------------
# ROUTES
# -----------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "audio_data" not in request.files:
            return jsonify({"emotion": None, "error": "audio_data missing"}), 400

        audio_file = request.files["audio_data"]

        filename = "rec_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio_file.save(filepath)

        emotion, probabilities = predict_emotion(filepath)
        print('emotion',emotion,'probabilities',probabilities)
        return jsonify({"emotion": emotion})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"emotion": None, "error": str(e)}), 500



# -----------------------------------------
# RUN SERVER
# -----------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
