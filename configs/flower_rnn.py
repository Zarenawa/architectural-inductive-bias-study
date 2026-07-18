import os
import torch

# PROJECT_ROOT = r'C:\Users\aliyu\Desktop\architectural-inductive-bias-study'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'dataset': 'flowers',
    'model_name': 'rnn',       # only this changed from flower_cnn.py
    'num_classes': 5,

    # Core training params — kept identical to CNN config, deliberately,
    # to isolate the architectural effect for the comparison (per your
    # earlier decision on same hyperparameters across all 3 models)
    'batch_size': 64,
    'epochs': 40,
    'lr': 1e-3,
    'input_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project_root': PROJECT_ROOT,
    'train_dir': os.path.join(PROJECT_ROOT, 'datasets/flowers/train'),
    'val_dir': os.path.join(PROJECT_ROOT, 'datasets/flowers/val'),
    'test_dir': os.path.join(PROJECT_ROOT, 'datasets/flowers/test'),

    # Single-phase training
    'phase1_augmentation': 'light',

    # Optimizer settings
    'weight_decay': 5e-4,
    'dropout_rate': 0.3,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_eps': 1e-8,

    # Regularization
    'label_smoothing': 0.1,

    # Training control
    'early_stopping_patience': 6,

    # --- RNN-specific architecture parameters ---
    # These aren't optimization hyperparameters (like lr/batch_size), so
    # they don't violate the "keep hyperparameters identical" principle —
    # they're structural choices forced by RNN's architecture, similar to
    # how CNN needed its own conv channel counts (32, 64) that MLP/RNN don't share
    'rnn_hidden_size': 128,     # size of the RNN's hidden state per timestep
    'rnn_num_layers': 1,        # stacked RNN layers
    'rnn_type': 'lstm',         # 'rnn', 'lstm', or 'gru' — LSTM recommended, since
                                 # plain RNN is known to suffer badly from vanishing/
                                 # exploding gradients over 128 timesteps (one per image row)
}