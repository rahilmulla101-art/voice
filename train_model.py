import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# -----------------------------
# Parameters
# -----------------------------
AUDIO_DIR = "data/english/"  # Root folder containing emotion subfolders
CSV_PATH = "features.csv"
MAX_DURATION = 3  # seconds

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path, max_duration=MAX_DURATION):
    y, sr = librosa.load(file_path, sr=None)

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
    return features

# -----------------------------
# Load Dataset and Extract Features
# -----------------------------
print("\n📥 Extracting features from audio files...")

data_list = []
for emotion_folder in os.listdir(AUDIO_DIR):
    emotion_path = os.path.join(AUDIO_DIR, emotion_folder)
    if not os.path.isdir(emotion_path):
        continue  # skip non-folders

    wav_files = [f for f in os.listdir(emotion_path) if f.endswith(".wav")]
    if not wav_files:
        print(f"⚠️  No .wav files found in {emotion_path}, skipping...")
        continue

    for wav_file in wav_files:
        file_path = os.path.join(emotion_path, wav_file)
        try:
            features = extract_features(file_path)
            data_list.append(np.append(features, emotion_folder))
        except Exception as e:
            print(f"⚠️  Failed to process {file_path}: {e}")

# Safety check
if not data_list:
    raise ValueError(f"No audio features were extracted! Check {AUDIO_DIR}.")

# -----------------------------
# Create DataFrame
# -----------------------------
feature_names = [f"f{i}" for i in range(len(data_list[0]) - 1)]
df = pd.DataFrame(data_list, columns=feature_names + ["label"])
df.to_csv(CSV_PATH, index=False)
print(f"💾 Features saved to {CSV_PATH}")

# -----------------------------
# Prepare Data for Training
# -----------------------------
X = df.drop("label", axis=1).values.astype(float)
y = df["label"].values

# Encode labels
print("🔤 Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("💾 Label encoder saved → label_encoder.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential([
    Dense(256, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train Model
# -----------------------------
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_split=0.2
)

# -----------------------------
# Save Model & Classes
# -----------------------------
model.save("emotion_model.h5")
np.save("classes.npy", label_encoder.classes_)
print("\n🎉 Training complete!")
print("💾 Model saved → emotion_model.h5")
print("💾 Classes saved → classes.npy")
