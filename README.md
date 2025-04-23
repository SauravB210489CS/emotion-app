# 🎤 Real-Time Emotion Detection App with LMAC Interpretability

This is a real-time speech emotion detection system powered by deep learning and Streamlit. It records audio from your microphone, predicts the emotion using a trained CNN+LSTM model, and visualizes the time regions of audio that influenced the prediction using LMAC (Local Model-Agnostic Classification).

---

## 🚀 Features

- 🎙️ Records live audio via microphone
- 🧠 Predicts emotion using a CNN + LSTM model
- 📊 Displays log-mel spectrogram
- 🔍 Shows LMAC-based interpretability of predictions
- 🌐 Deployable on Streamlit Cloud

---

## 📦 Requirements

- Python 3.7+
- torch
- librosa
- sounddevice
- scipy
- matplotlib
- joblib
- streamlit
- gdown

Install all with:

```bash
pip install -r requirements.txt

---
Link to Dataset : https://drive.google.com/drive/folders/1Qjmc_kHq9i2lDXhtorzmNh5u_Gb6gEWT?usp=drive_link
