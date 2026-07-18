import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        num_classes = config.get('num_classes', 5) if config else 5
        input_size = config.get('input_size', 128) if config else 128
        in_channels = config.get('in_channels', 3) if config else 3
        dropout = config.get('dropout_rate', 0.2) if config else 0.2

        hidden_size = config.get('rnn_hidden_size', 128) if config else 128
        num_layers = config.get('rnn_num_layers', 1) if config else 2
        rnn_type = config.get('rnn_type', 'lstm') if config else 'lstm'

        # Image is treated as a sequence of rows: each row (across all
        # channels) becomes one timestep. So sequence_length = input_size (H),
        # and each timestep's feature vector = in_channels * input_size (C*W).
        # This row-as-timestep approach follows the same idea used in
        # PixelRNN's Row LSTM (van den Oord et al.), applying a sequential
        # model to image data despite images having no natural temporal order.
        self.sequence_length = input_size
        self.feature_per_step = in_channels * input_size

        # Select RNN variant — LSTM/GRU recommended over plain RNN for longer
        # sequences (e.g. input_size=128 means 128 timesteps), since plain RNN
        # is known to suffer from vanishing/exploding gradients over long sequences
        rnn_type = rnn_type.lower()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.feature_per_step,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.feature_per_step,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=self.feature_per_step,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Use 'rnn', 'lstm', or 'gru'.")

        self.rnn_type = rnn_type

        # Classifier head — same shape/style as CNN's, for a fair architectural
        # comparison (one hidden Linear layer, ReLU, Dropout, output Linear)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x arrives as (B, C, H, W) — same format the CNN expects.
        # Reshape: (B, C, H, W) -> (B, H, C, W) -> (B, H, C*W)
        # so each of the H rows becomes one timestep, with all channels'
        # pixel values for that row flattened into the feature vector.
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 1, 3)              # (B, H, C, W)
        x = x.reshape(batch_size, height, channels * width)  # (B, H, C*W)

        if self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(x)
        else:
            _, hidden = self.rnn(x)

        # Use the final layer's hidden state as the sequence summary —
        # hidden shape is (num_layers, B, hidden_size); take the last layer
        last_hidden = hidden[-1]  # (B, hidden_size)

        out = self.classifier(last_hidden)
        return out