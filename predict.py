import librosa
import numpy as np
import tensorflow as tf
import pickle

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "emotion_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# -----------------------------
# Load Model & Label Encoder
# -----------------------------
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print("✅ Model and label encoder loaded successfully!")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path, max_duration=3):
    y, sr = librosa.load(file_path, sr=None)
    print(f"Audio loaded: {file_path}, length: {len(y)}, sample rate: {sr}")

    # Normalize audio length
    max_len = sr * max_duration
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Combine all features
    features = np.concatenate([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(tonnetz.T, axis=0)
    ])
    
    print(f"Feature vector shape: {features.shape}")
    return features

# -----------------------------
# Emotion Prediction
# -----------------------------
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features_exp = np.expand_dims(features, axis=0)  # reshape for model

    # Model prediction
    predictions = model.predict(features_exp)
    print("Raw model predictions (probabilities):", predictions)

    # Predicted emotion
    class_index = np.argmax(predictions)
    emotion = le.inverse_transform([class_index])[0]
    print(f"Predicted class index: {class_index}")
    print(f"Predicted Emotion: {emotion}")

    # Optional: print probability of each emotion
    for idx, prob in enumerate(predictions[0]):
        print(f"{le.classes_[idx]}: {prob:.3f}")

    return emotion

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    audio_file = "test1.wav"  # change this to your audio file
    print("\n🎯 Predicting emotion for:", audio_file)
    result = predict_emotion(audio_file)
    print("\n✅ Final Predicted Emotion:", result)
