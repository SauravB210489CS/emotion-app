import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from model_cnn_lstm import CNNLSTMEmotionModel
import joblib

# Paths
WAV_PATH = os.path.expanduser("~/Desktop/finalyearproject/live_input.wav")
MODEL_PATH = os.path.expanduser("~/Desktop/finalyearproject/best_emotion_model.pt")
ENCODER_PATH = os.path.expanduser("~/Desktop/finalyearproject/label_encoder.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
le = joblib.load(ENCODER_PATH)
model = CNNLSTMEmotionModel(num_classes=len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load audio
y, sr = librosa.load(WAV_PATH, sr=16000)
if len(y) < 48000:
    y = np.pad(y, (0, 48000 - len(y)))
else:
    y = y[:48000]

# Base prediction
def get_prediction(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        output = model(x)
        return torch.softmax(output, dim=1).cpu().numpy()[0]

baseline_pred = get_prediction(y)
predicted_class = np.argmax(baseline_pred)
print(f"🎯 Base prediction: {le.inverse_transform([predicted_class])[0]}")

# LMAC: Sliding mask
window_size = int(0.2 * sr)  # 200ms window
stride = int(0.1 * sr)
importance = []

for start in range(0, len(y), stride):
    end = min(start + window_size, len(y))
    masked = np.copy(y)
    masked[start:end] = 0.0
    new_pred = get_prediction(masked)
    diff = baseline_pred[predicted_class] - new_pred[predicted_class]
    importance.append(diff)

# Normalize & pad to full audio length
importance = np.array(importance)
importance = np.pad(importance, (0, (len(y) - len(importance * stride)) // stride + 1))

# Plot result
plt.figure(figsize=(12, 4))
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
log_mel = librosa.power_to_db(mel, ref=np.max)
librosa.display.specshow(log_mel, sr=sr, x_axis="time", y_axis="mel")
plt.title(f"LMAC Interpretability — {le.inverse_transform([predicted_class])[0]}")
plt.plot(np.linspace(0, log_mel.shape[1], len(importance)), importance * 100, color="red", linewidth=2)
plt.tight_layout()
plt.show()
