import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        num_classes = config.get('num_classes', 5) if config else 5
        input_size = config.get('input_size', 128) if config else 128
        in_channels = config.get('in_channels', 3) if config else 3
        dropout = config.get('dropout_rate', 0.2) if config else 0.2
        hidden_sizes = config.get('mlp_hidden_sizes', [128, 64]) if config else [128,64]

        # MLP has no spatial structure at all — the input image gets
        # completely flattened into a single vector before anything else
        # happens. This is the key architectural difference from CNN
        # (which preserves 2D structure via convolution) and RNN (which
        # preserves row-order structure via sequential processing).
        flattened_input_size = in_channels * input_size * input_size

        # Build hidden layers dynamically from the config list, so changing
        # mlp_hidden_sizes doesn't require editing this file
        layers = []
        prev_size = flattened_input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Final classification layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            *layers
        )

    def forward(self, x):
        # x arrives as (B, C, H, W) — same format CNN and RNN expect.
        # nn.Flatten() inside self.classifier handles the actual flattening.
        x = self.classifier(x)
        return x