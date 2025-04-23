import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import joblib
import os
from scipy.io.wavfile import write
from model_cnn_lstm import CNNLSTMEmotionModel

# Config
SAMPLE_RATE = 16000
DURATION = 3
WAV_PATH = os.path.expanduser("~/Desktop/finalyearproject/live_input.wav")
MODEL_PATH = os.path.expanduser("~/Desktop/finalyearproject/best_emotion_model.pt")
ENCODER_PATH = os.path.expanduser("~/Desktop/finalyearproject/label_encoder.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and label encoder
@st.cache_resource
def load_model():
    model = CNNLSTMEmotionModel(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)

label_encoder = load_encoder()
model = load_model()

# Record Audio
def record_audio():
    st.info("Recording for 3 seconds...")
    recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(WAV_PATH, SAMPLE_RATE, recording)
    st.success("Recording complete.")
    return WAV_PATH

# Predict Emotion
def predict_emotion(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(prob)
    return label_encoder.inverse_transform([pred_idx])[0], prob

# LMAC-style interpretation
def lmac_importance(y, baseline_prob):
    stride = int(0.1 * SAMPLE_RATE)
    win = int(0.2 * SAMPLE_RATE)
    pred_idx = np.argmax(baseline_prob)
    importance = []

    for start in range(0, len(y), stride):
        end = min(start + win, len(y))
        y_masked = y.copy()
        y_masked[start:end] = 0.0
        _, prob_masked = predict_emotion(y_masked)
        diff = baseline_prob[pred_idx] - prob_masked[pred_idx]
        importance.append(diff)

    return np.array(importance)

# Streamlit UI
st.title("🎤 Real-Time Emotion Detection with LMAC")
if st.button("🎙️ Record & Predict"):
    wav_path = record_audio()
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
    else:
        y = y[:SAMPLE_RATE * DURATION]

    st.audio(wav_path, format='audio/wav')
    pred, prob = predict_emotion(y)
    st.subheader(f"🔍 Predicted Emotion: `{pred}`")

    # Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=ax)
    ax.set(title="Log-Mel Spectrogram")
    st.pyplot(fig)

    # LMAC Visualization
    st.subheader("📊 LMAC Interpretability")
    importance = lmac_importance(y, prob)
    time_steps = np.linspace(0, DURATION, len(importance))
    fig2, ax2 = plt.subplots()
    ax2.plot(time_steps, importance * 100, color="red", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Importance (%)")
    ax2.set_title("Audio Segment Importance (LMAC)")
    st.pyplot(fig2)
