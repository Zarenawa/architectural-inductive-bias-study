import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        num_classes = config.get('num_classes', 5) if config else 5
        input_size = config.get('input_size', 128) if config else 128
        in_channels = config.get('in_channels', 3) if config else 3
        dropout = config.get('dropout_rate', 0.2) if config else 0.2

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Global Average Pooling: collapses each channel's H×W map
            # down to a single value, regardless of input_size.
            # This is the key fix — instead of flattening 64×32×32=65,536
            # values into a huge Linear layer, we now only pass 64 values
            # into the classifier, cutting ~8.3M parameters down to ~8K.
            nn.AdaptiveAvgPool2d(1)
        )

        # With GAP, flattened_size is always just the channel count (64) —
        # no longer depends on input_size at all, but keeping the dummy
        # forward pass anyway so this stays robust if you add more conv layers later
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            dummy_out = self.features(dummy)
            flattened_size = dummy_out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x