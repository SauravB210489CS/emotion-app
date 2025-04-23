import os
import torch
import librosa
import numpy as np
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
from model_cnn_lstm import CNNLSTMEmotionModel

# Paths
MODEL_PATH = os.path.expanduser("~/Desktop/finalyearproject/best_emotion_model.pt")
ENCODER_PATH = os.path.expanduser("~/Desktop/finalyearproject/label_encoder.pkl")
TEMP_WAV = os.path.expanduser("~/Desktop/finalyearproject/live_input.wav")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load label encoder and model
le = joblib.load(ENCODER_PATH)
model = CNNLSTMEmotionModel(num_classes=len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Record from microphone
def record_audio(filename, duration=3, samplerate=16000):
    print(f"\n🎙️ Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, recording)
    print(f"✅ Recording saved: {filename}")

# Predict emotion from .wav file
def predict_emotion(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    if len(y) < 48000:
        y = np.pad(y, (0, 48000 - len(y)))
    else:
        y = y[:48000]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        out = model(x)
        pred_class = torch.argmax(out, dim=1).item()
        pred_label = le.inverse_transform([pred_class])[0]
        print(f"\n🎧 Predicted Emotion: **{pred_label}**\n")

# Main trigger
if __name__ == "__main__":
    record_audio(TEMP_WAV)
    predict_emotion(TEMP_WAV)

