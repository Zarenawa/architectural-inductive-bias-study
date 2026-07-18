"""
Flowers Image DataLoader (Supports CNN, MLP, RNN)

Purpose:
Creates PyTorch dataloaders for flower classification (train/val/test).

Features:
- Shared across CNN, MLP, RNN — dataloading is architecture-agnostic;
  any reshaping (flatten for MLP, sequence for RNN) happens inside each
  model's forward(), not here.
- Config-driven, same override pattern as the original bird dataloader.
- Simplified augmentation set: 'none', 'light', 'randaugment' — the
  presets actually used for this project. 'strong' and 'autoaugment'
  were dropped since they weren't needed here, but the structure is
  left open if you want to add one back later.
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List
from torchvision.transforms import RandAugment

def get_dataloaders(
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    augmentation: str = 'light',
    config: Optional[dict] = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    Creates PyTorch DataLoaders for whichever splits are provided.
    Pass only the directories you need — e.g. test_dir alone for evaluation-only use.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    if config:
        train_dir = config.get('train_dir', train_dir)
        val_dir = config.get('val_dir', val_dir)
        test_dir = config.get('test_dir', test_dir)
        batch_size = config.get('batch_size', batch_size)
        augmentation = config.get('phase1_augmentation', augmentation)

    if batch_size is None:
        raise ValueError("batch_size must be specified in get_dataloaders")
    if train_dir is None and val_dir is None and test_dir is None:
        raise ValueError("At least one of train_dir, val_dir, or test_dir must be specified")

    input_size = config.get('input_size', 128) if config else 128

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    if augmentation == 'light':
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation == 'randaugment':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError(f"Invalid augmentation preset: {augmentation}")

    eval_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    class_names = None

    if train_dir is not None:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)
        class_names = train_dataset.classes
    else:
        train_loader = None

    if val_dir is not None:
        val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
        if class_names is None:
            class_names = val_dataset.classes
    else:
        val_loader = None

    if test_dir is not None:
        test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)
        if class_names is None:
            class_names = test_dataset.classes
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, class_names