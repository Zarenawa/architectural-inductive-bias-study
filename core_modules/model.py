"""
Modular Model Builder (CNN, MLP, RNN)

Supports:
- CNN  (core_modules/cnn.py)
- MLP  (core_modules/mlp.py)
- RNN  (core_modules/rnn.py)

This file only dispatches to the right architecture based on config['model_name'].
The actual model definitions live in core_modules/, same pattern as your
original EfficientNet builder.
"""

import torch
from torch.nn import Module
from typing import Optional, Dict


def build_model(
    device: str = 'cpu',
    checkpoint_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Module:
    """
    Initializes a CNN, MLP, or RNN model based on config['model_name'].

    Args:
        device: 'cuda' or 'cpu'
        checkpoint_path: Optional path to a saved state_dict, for resuming training
                         or loading a trained model for evaluation
        config: Config dict — must include 'model_name' and 'num_classes'

    Returns:
        Configured model, moved to device
    """
    if config:
        device = config.get('device', device)
        model_name = config.get('model_name')
    else:
        model_name = None

    if model_name is None:
        raise ValueError("model_name must be specified in config")

    # CNN — spatial convolutions, expects (B, C, H, W) input
    if model_name == 'cnn':
        from core_modules.cnn import CNN
        print("Model: CNN")
        model = CNN(config=config)

    # MLP — fully-connected, expects flattened input; flattening happens
    # inside MLP's own forward(), not here
    elif model_name == 'mlp':
        from core_modules.mlp import MLP
        print("Model: MLP")
        model = MLP(config=config)

    # RNN — sequential, treats image rows as timesteps; reshaping happens
    # inside RNN's own forward(), not here
    elif model_name == 'rnn':
        from core_modules.rnn import RNN
        print("Model: RNN")
        model = RNN(config=config)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Optional: load weights from a checkpoint (resuming training, or loading
    # a trained model for evaluation later)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle both possible saved formats:
        # 1. Raw weights only: torch.save(model.state_dict(), path)
        # 2. Wrapped dict: torch.save({'state_dict': ..., 'class_names': ..., 'config': ...}, path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Print summary after build
    print(model_summary(model))

    return model.to(device)


def model_summary(model: Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if hasattr(model, 'classifier'):
        head = model.classifier
    elif hasattr(model, 'fc'):
        head = model.fc
    else:
        head = "Unknown"

    return (
        f"Model Configuration:\n"
        f"├── Trainable params: {trainable:,}\n"
        f"├── Frozen params: {frozen:,}\n"
        f"└── Output head: {head}\n"
    )