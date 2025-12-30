"""
Módulo para carregar e gerenciar configurações de experimentos via YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega arquivo de configuração YAML.
    
    Args:
        config_path: Caminho para arquivo .yaml
        
    Returns:
        Dicionário com configurações
        
    Example:
        >>> config = load_config('config/resnet50_config.yaml')
        >>> print(config['training']['learning_rate'])
        0.0001
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuração carregada de: {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Salva configuração em arquivo YAML.
    
    Args:
        config: Dicionário com configurações
        save_path: Caminho onde salvar
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Configuração salva em: {save_path}")


# Configuração padrão (template)
DEFAULT_CONFIG = {
    'model': {
        'name': 'resnet50',  # ou 'efficientnet_b0'
        'pretrained': True,
        'num_classes': 2
    },
    'data': {
        'dataset': 'isic2020',
        'data_dir': './data/isic2020',
        'batch_size': 32,
        'num_workers': 4,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'image_size': 224
    },
    'training': {
        'optimizer': 'adam',
        'learning_rate': 0.0001,
        'weight_decay': 0.00001,
        'epochs': 50,
        'early_stopping_patience': 10,
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-7
    },
    'loss': {
        'type': 'weighted_cross_entropy',
        'class_weights': [1.0, 56.0]  # [benigno, maligno]
    },
    'augmentation': {
        'rotation': 30,
        'horizontal_flip': 0.5,
        'vertical_flip': 0.5,
        'brightness': 0.2,
        'contrast': 0.2,
        'zoom_range': [0.8, 1.2]
    },
    'random_seed': 42,
    'device': 'cuda',
    'checkpoint_dir': './checkpoints',
    'log_dir': './runs'
}