"""
Shared TCN1D CNN architecture for HINTSight vHIT impulse classification.

This must stay identical to the model trained in HITClassification.ipynb so
that the pretrained weights in cnn4_tcn_best.pt load correctly. Both
train_fusion_model.py and the live inference pipeline import from here so the
architecture is defined in exactly one place.
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out + residual
        return self.final_relu(out)


class TCN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            TemporalBlock(2, 32, 3, dilation=1, dropout=0.10),
            TemporalBlock(32, 64, 3, dilation=2, dropout=0.10),
            TemporalBlock(64, 128, 3, dilation=4, dropout=0.15),
            TemporalBlock(128, 128, 3, dilation=8, dropout=0.15),
            TemporalBlock(128, 256, 3, dilation=4, dropout=0.15),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x):
        """Returns the 256-d pooled embedding concatenated with the 4-class softmax probs."""
        feat = self.network(x)
        pooled = self.global_pool(feat).flatten(1)
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=1)
        return torch.cat([pooled, probs], dim=1)

    def forward(self, x):
        x = self.network(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def load_cnn(checkpoint_path="cnn4_tcn_best.pt", device="cpu"):
    """Loads the frozen pretrained TCN1D along with its normalization stats and class names."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])

    model = TCN1D(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, checkpoint["mean"], checkpoint["std"], classes
