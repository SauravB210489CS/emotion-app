import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_cnn_lstm import CNNLSTMEmotionModel
from data_preprocessing import preprocess_dataset
from collections import Counter
import joblib

# Paths
BASE_DIR = os.path.expanduser("~/Desktop/finalyearproject/My Dataset")
CSV_PATH = os.path.expanduser("~/Desktop/finalyearproject/emotions.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Step 1: Load and preprocess dataset
X, y = preprocess_dataset(BASE_DIR, CSV_PATH)

# Step 2: Clean labels with less than 2 samples
label_counts = Counter(y)
valid_labels = [label for label, count in label_counts.items() if count >= 2]
mask = np.isin(y, valid_labels)
X = X[mask]
y = y[mask]

# Step 3: Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Step 5: Define custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1).float()  # shape: [batch, 1, mel, time]
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(EmotionDataset(X_test, y_test), batch_size=BATCH_SIZE)

# Step 6: Model, Loss, Optimizer
model = CNNLSTMEmotionModel(num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 7: Training Loop
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Evaluation
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(train_losses):.4f} | Accuracy: {acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.expanduser("~/Desktop/finalyearproject/best_emotion_model.pt"))
        print("💾 Best model saved.")

# Save LabelEncoder
joblib.dump(le, os.path.expanduser("~/Desktop/finalyearproject/label_encoder.pkl"))

