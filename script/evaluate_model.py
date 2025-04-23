import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from model_cnn_lstm import CNNLSTMEmotionModel
from data_preprocessing import preprocess_dataset

# Load model and label encoder
MODEL_PATH = os.path.expanduser("~/Desktop/finalyearproject/best_emotion_model.pt")
ENCODER_PATH = os.path.expanduser("~/Desktop/finalyearproject/label_encoder.pkl")
BASE_DIR = os.path.expanduser("~/Desktop/finalyearproject/My Dataset")
CSV_PATH = os.path.expanduser("~/Desktop/finalyearproject/emotions.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load and preprocess data
X, y = preprocess_dataset(BASE_DIR, CSV_PATH)
le = joblib.load(ENCODER_PATH)

from collections import Counter
from sklearn.model_selection import train_test_split

valid_labels = [label for label, count in Counter(y).items() if count >= 2]
mask = np.isin(y, valid_labels)
X = X[mask]
y = y[mask]
y_encoded = le.transform(y)

_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1).float()
        self.y = torch.tensor(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_loader = DataLoader(EmotionDataset(X_test, y_test), batch_size=16)

# Load model
model = CNNLSTMEmotionModel(num_classes=len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Report
print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/finalyearproject/confusion_matrix_eval.png"))
plt.show()

