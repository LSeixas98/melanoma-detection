"""
Módulo para transformações de dados e augmentações.
"""

from typing import Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import WeightedRandomSampler


# Normalização padrão do ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(config: Dict[str, Any], train: bool = True) -> A.Compose:
    """
    Retorna transformações para treino ou validação/teste.
    
    Args:
        config: Dicionário de configuração com chaves 'data' e 'augmentation'
        train: Se True, aplica augmentações; senão, apenas resize e normalização
    """
    image_size = config['data'].get('image_size', 224)
    aug_config = config.get('augmentation', {})
    
    if train:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=aug_config.get('rotation', 30), p=0.5),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=aug_config.get('vertical_flip', 0.5)),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness', 0.2),
                contrast_limit=aug_config.get('contrast', 0.2),
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=aug_config.get('zoom_range', [0.8, 1.2]),
                rotate_limit=0,
                p=0.5
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    
    return transform


def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Cria sampler balanceado para lidar com classes desbalanceadas.
    
    Calcula pesos inversamente proporcionais à frequência de cada classe.
    """
    targets = dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
