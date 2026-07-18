import os
import torch

# PROJECT_ROOT = r'C:\Users\aliyu\Desktop\architectural-inductive-bias-study'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'dataset': 'flowers',
    'model_name': 'mlp',       # only this differs from flower_cnn.py / flower_rnn.py
    'num_classes': 5,

    # Core training params — kept identical to CNN/RNN configs, deliberately,
    # to isolate the architectural effect for the comparison
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

    # --- MLP-specific architecture parameters ---
    # Structural choices, not optimization hyperparameters — same reasoning
    # as rnn_hidden_size/rnn_num_layers not violating the "identical
    # hyperparameters" principle, since CNN also had its own architecture-only
    # values (32/64 conv channels) that don't apply to MLP or RNN either
    'mlp_hidden_sizes': [128, 64],  # sizes of MLP's hidden layers, in order
}