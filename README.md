# 🎙️ Real-Time Speech Emotion Recognition with Explainability

This project implements a robust **Speech Emotion Recognition (SER)** system using deep learning techniques. It is designed for real-time prediction from live or uploaded audio. The model is based on a **CNN-BiLSTM-Attention** architecture and provides **explainability** using **Local Model-Agnostic Classification (LMAC)**, helping users understand *why* a particular emotion was predicted.

---

## 🔍 What This Project Does

This system detects emotions such as **happy**, **sad**, **angry**, **neutral**, etc., from speech signals by:

1. **Extracting rich acoustic features** (MFCC, Mel, Chroma, Spectral Contrast, Tonnetz)
2. **Training a hybrid deep learning model** (CNN + BiLSTM + Attention)
3. **Deploying a Streamlit web app** for real-time audio input and prediction
4. **Visualizing predictions with waveform, spectrogram, and explainability**
5. **Improving interpretability using LMAC** to highlight top contributing features

---

## 📁 Project Structure

finalyearproject/
│
├── My Dataset/ # Structured dataset of .wav files across emotions
├── features/ # Preprocessed 180-dim features
├── models/ # Trained PyTorch model weights (.pt)
├── scaler_encoder/ # Sklearn StandardScaler and LabelEncoder
├── screenshots/ # App UI and explanation visuals
├── augmented_data/ # Pitch, noise, and time-augmented audio
├── app.py # Streamlit UI with real-time prediction + explanation
├── train_model.py # CNN-BiLSTM-Attention training loop
├── utils.py # Helper functions for feature extraction, LMAC, plotting
├── spectrogram_plot.py # Spectrogram drawing during recording/upload
├── requirements.txt # All dependencies
└── README.md # You're reading it!


---

## 🎯 Goals Achieved

✅ High-quality real-time SER pipeline  
✅ Trained deep neural network with >80% accuracy  
✅ Fully integrated Streamlit interface  
✅ LMAC-based interpretability of predictions  
✅ Audio visualization (waveform + spectrogram + timer)  
✅ Data augmentation to increase robustness  
✅ Live inference for both recorded and uploaded inputs

---

## 🛠️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# (Optional) Train the model from scratch
python train_model.py

# Run the Streamlit app
streamlit run app.py
Model Overview
Component	Purpose
CNN Layers	Extract local patterns from temporal features (e.g., MFCC time series)
BiLSTM Layer	Capture long-range temporal dependencies across time steps
Attention	Focus on emotionally salient frames
Dense Layer	Final emotion classification (8 classes)

Input Shape: (180,)

Output Classes: ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sarcastic', 'surprise']

📈 Feature Extraction
Extracted using Librosa:

MFCC (40 coefficients)

Mel Spectrogram (40)

Chroma Features (12)

Spectral Contrast (7)

Tonnetz (6)

All features are mean-pooled over time, resulting in a 180-dimensional input vector per audio file.

🔄 Data Augmentation Techniques
To improve generalization and reduce overfitting:

✅ Pitch shifting

✅ Time stretching

✅ Noise injection

Each raw audio sample is transformed into multiple variations to simulate real-world variability.

🧪 Training Details
Parameter	Value
Optimizer	Adam
Learning Rate	0.0005
Batch Size	32
Epochs	100
Loss Function	CrossEntropy
Validation Split	20%

The model is trained with live training and validation accuracy/loss plots to monitor convergence.

📊 Evaluation
✅ Accuracy > 80% on validation set

✅ Confusion matrix analysis

✅ Per-class accuracy visualization

✅ LMAC explanations for every prediction

✅ Confidence score shown in real-time interface

🔍 Explainability with LMAC
Local Model-Agnostic Classification (LMAC) explains predictions by:

Perturbing inputs and observing model changes

Ranking top contributing features (MFCC, Mel, etc.)

Displaying a bar chart showing feature importance per prediction

Helps users understand why the model predicted a specific emotion.

🌐 Streamlit Interface Features
🎙️ Start Recording: Capture 10 seconds of audio

📂 Upload Audio: Use existing .wav file

📉 Real-time Spectrogram + Waveform

📊 Predicted Emotion + Confidence

🧠 LMAC bar graph for interpretability

⏳ Countdown timer during recording

📷 Sample Visuals
Spectrogram	LMAC Explainability

📌 Future Improvements
Use transformer-based models (e.g., Wav2Vec 2.0)

Add emotion localization within audio

Create a mobile version of the app

Add speaker recognition and emotion intensity scoring

👏 Acknowledgments
📘 LMAC - Francesco Paissan

🔧 Librosa, PyTorch, Streamlit, Scikit-learn

📜 License
This project is licensed under the MIT License. Feel free to fork and contribute!
