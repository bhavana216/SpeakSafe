import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

REAL_DIR = 'dataset-real'
FAKE_DIR = 'dataset-fake'

def extract_features(file_path, sr=16000, n_mfcc=13):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

features = []
labels = []


for filename in os.listdir(REAL_DIR):
    if filename.endswith('.wav'):
        path = os.path.join(REAL_DIR, filename)
        mfcc = extract_features(path)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(0)


for filename in os.listdir(FAKE_DIR):
    if filename.endswith('.wav'):
        path = os.path.join(FAKE_DIR, filename)
        mfcc = extract_features(path)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(1)


X = np.array(features)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, 'audio_classification_model.pkl')
print("✅ Model trained and saved as audio_classification_model.pkl")

