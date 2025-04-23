import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Settings
SAMPLE_RATE = 16000
DURATION = 3  # seconds
OUTPUT_PATH = os.path.expanduser("~/Desktop/finalyearproject/live_input.wav")

def record_audio(path):
    print("🎙️ Recording for 3 seconds...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(path, SAMPLE_RATE, recording)
    print(f"✅ Audio saved: {path}")

def show_spectrogram(path):
    print("📊 Generating spectrogram...")
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
    else:
        y = y[:SAMPLE_RATE * DURATION]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Live Log-Mel Spectrogram")
    plt.tight_layout()
    plt.show()

# Run everything
if __name__ == "__main__":
    record_audio(OUTPUT_PATH)
    show_spectrogram(OUTPUT_PATH)
