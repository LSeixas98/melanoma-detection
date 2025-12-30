"""
Factory para criação de modelos baseado em configuração.
"""

import torch.nn as nn
from models.resnet import get_resnet50
from models.efficientnet import get_efficientnet_b0


def create_model(config: dict) -> nn.Module:
    """
    Cria modelo baseado em configuração.
    
    Args:
        config: Dicionário de configuração com chave 'model'
        
    Returns:
        Instância do modelo
        
    Raises:
        ValueError: Se nome do modelo não for reconhecido
        
    Example:
        >>> config = {'model': {'name': 'resnet50', 'pretrained': True, 'num_classes': 2}}
        >>> model = create_model(config)
    """
    model_config = config['model']
    model_name = model_config['name'].lower()
    num_classes = model_config.get('num_classes', 2)
    pretrained = model_config.get('pretrained', True)
    
    if model_name == 'resnet50':
        model = get_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. "
            f"Opções disponíveis: 'resnet50', 'efficientnet_b0'"
        )
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Retorna informações detalhadas sobre o modelo.
    
    Args:
        model: Instância do modelo PyTorch
        
    Returns:
        Dicionário com informações (parâmetros, nome, etc.)
    """
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Nome do modelo
    model_name = model.__class__.__name__
    
    info = {
        'name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'size_mb': total_params * 4 / (1024 ** 2)  # Assumindo float32 (4 bytes)
    }
    
    return info