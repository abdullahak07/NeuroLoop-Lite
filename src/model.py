"""
model.py -- one small 1D CNN.

EEG channels are treated as input feature channels; the convolutions run over
time. Deliberately small for a portfolio demo. No transformers or ensembles.
"""

import torch
import torch.nn as nn


class TinyEEGCNN(nn.Module):
    def __init__(self, n_channels, n_classes=2, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=7, padding=3),
                nn.BatchNorm1d(cout),
                nn.ELU(),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(n_channels, 32),
            block(32, 64),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(n_channels, n_classes=2):
    return TinyEEGCNN(n_channels, n_classes)
