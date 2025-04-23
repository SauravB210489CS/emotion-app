import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTMEmotionModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # reduces mel from 64 → 32

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))   # reduces mel from 32 → 16
        )

        # CNN output shape will be [batch, 64, 16, time']
        # Flattened last two dims → 64 * 16 = 1024
        self.projector = nn.Linear(1024, 64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, 1, mel, time]
        x = self.cnn(x)              # → [batch, 64, 16, time']
        x = x.permute(0, 3, 1, 2)    # → [batch, time', channels, mel']
        b, t, c, m = x.shape
        x = x.reshape(b, t, c * m)   # → [batch, time', 1024]
        x = self.projector(x)        # → [batch, time', 64]
        lstm_out, _ = self.lstm(x)   # → [batch, time', 128]
        out = self.classifier(lstm_out[:, -1, :])  # last time step → [batch, num_classes]
        return out

