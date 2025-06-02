# 🎙️ SpeakSafe: Deepfake Audio Detection Tool

SpeakSafe is an intelligent and interactive web app designed to detect deepfake audio using machine learning. Built with Python and Streamlit, it offers a clean, colorful UI and intuitive features that help determine whether an audio clip is genuine or fake.

## 🚀 Features

* 🔊 **Upload Audio or Record via Microphone**
* 🔝 Real-time **deepfake prediction with confidence scores**
* 📈 ML-based model trained on real vs fake audio samples
* 🌟 Clean, colorful and animated UI
* 🌐 Works locally or on Streamlit Cloud

---

## 🎓 How It Works

1. **Feature Extraction**: Converts audio to MFCC features.
2. **Model Prediction**: Uses a trained classifier to evaluate features.
3. **Result**: Displays whether the audio is Real or Fake with a confidence percentage.

---


## 🔧 Project Structure

```
SpeakSafe/
├── app.py                  # Streamlit app
├── utils.py                # Feature extraction & model prediction
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── dataset-real/           # Folder with real audio
├── dataset-fake/           # Folder with fake audio
├── model/
│   └── audio_classifier.pkl  # Saved ML model
```

## 🤖 Tech Stack

* Python
* Streamlit
* Scikit-learn
* Librosa
* NumPy, Pandas
* pydub
* streamlit-webrtc


## 📅 Future Improvements

* Add support for longer audio files
* Include spectrogram visualizations
* Deploy to Hugging Face Spaces or Streamlit Cloud
* Add multilingual support for UI



## 🎉 Footer

> Made with ❤️ by **Bhavana** using **Streamlit** and **Machine Learning** 🚀
