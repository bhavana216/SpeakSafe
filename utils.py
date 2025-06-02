import librosa
import numpy as np
import joblib

def extract_features(file_path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Load the trained model from the correct filename
model = joblib.load("audio_classification_model.pkl")

def model_predict(file_path):
    features = extract_features(file_path)
    if features is not None:
        prediction = model.predict([features])[0]
        confidence = max(model.predict_proba([features])[0])
        return prediction, confidence
    else:
        return "Error", 0.0
