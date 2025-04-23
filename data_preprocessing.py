import os
import librosa
import numpy as np
import pandas as pd
import torch

# Constants
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 64
FIXED_LENGTH = SAMPLE_RATE * DURATION

def load_wav(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) < FIXED_LENGTH:
        y = np.pad(y, (0, FIXED_LENGTH - len(y)))
    else:
        y = y[:FIXED_LENGTH]
    return y

def extract_log_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def preprocess_dataset(base_dir, csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        file_path = os.path.join(base_dir, row['filename'])
        label = row['emotion'].strip().lower()

        try:
            audio = load_wav(file_path)
            log_mel = extract_log_mel(audio)
            X.append(log_mel)
            y.append(label)
            print(f"✅ Processed: {file_path}")
        except Exception as e:
            print(f"❌ Failed: {file_path} — {e}")

    return np.array(X), np.array(y)

