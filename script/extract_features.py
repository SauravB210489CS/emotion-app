import os
import librosa
import numpy as np
import pandas as pd

DATASET_DIR = os.path.expanduser("~/Desktop/finalyearproject/My Dataset")
OUTPUT_CSV = os.path.expanduser("~/Desktop/finalyearproject/features.csv")

features = []
labels = []

for actor in sorted(os.listdir(DATASET_DIR)):
    actor_path = os.path.join(DATASET_DIR, actor)
    if not os.path.isdir(actor_path): continue
    for session in sorted(os.listdir(actor_path)):
        session_path = os.path.join(actor_path, session)
        for emotion in sorted(os.listdir(session_path)):
            emotion_path = os.path.join(session_path, emotion)
            for file in os.listdir(emotion_path):
                if not file.endswith(".wav"):
                    continue
                file_path = os.path.join(emotion_path, file)
                try:
                    y, sr = librosa.load(file_path, sr=16000)
                    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
                    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
                    combined = np.concatenate([mfcc, chroma, mel])
                    features.append(combined)
                    labels.append(emotion)
                    print(f"✅ Processed: {file_path}")
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")

df = pd.DataFrame(features)
df["label"] = labels
df.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 Features saved to: {OUTPUT_CSV}")
