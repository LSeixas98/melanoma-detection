"""
Configuração padrão do projeto.
Centraliza todas as configurações para evitar duplicação.
"""

DEFAULT_CONFIG = {
    'data': {
        'data_dir': './data/isic2020',
        'batch_size': 32,
        'num_workers': 4,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'image_size': 224
    },
    'augmentation': {
        'rotation': 30,
        'horizontal_flip': 0.5,
        'vertical_flip': 0.5,
        'brightness': 0.2,
        'contrast': 0.2,
        'zoom_range': [0.8, 1.2]
    },
    'training': {
        'epochs': 50,
        'early_stopping_patience': 10,
        'learning_rate': 0.0001,
        'weight_decay': 0.00001,
        'optimizer': 'adam',
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-7
    },
    'loss': {
        'type': 'weighted_cross_entropy',
        'class_weights': [1.0, 56.0]  # [benigno, maligno]
    },
    'random_seed': 42
}


def get_config(model_name, data_dir=None, **overrides):
    """
    Retorna configuração para um modelo específico.
    
    Args:
        model_name: Nome do modelo ('resnet50' ou 'efficientnet_b0')
        data_dir: Diretório do dataset (opcional, sobrescreve default)
        **overrides: Parâmetros adicionais para sobrescrever defaults
    
    Returns:
        Dicionário de configuração completo
    """
    import copy
    
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    # Configurações específicas do modelo
    config['model'] = {'name': model_name}
    config['checkpoint_dir'] = f'./checkpoints/{model_name}'
    config['log_dir'] = './runs'
    config['device'] = 'cuda'  # Será detectado automaticamente
    
    # Sobrescrever data_dir se fornecido
    if data_dir:
        config['data']['data_dir'] = data_dir
    
    # Aplicar overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    return config


def get_training_config(model_name, data_dir=None, **overrides):
    """
    Retorna configuração otimizada para treinamento.
    Inclui todas as chaves necessárias para o Trainer.
    
    Args:
        model_name: Nome do modelo
        data_dir: Diretório do dataset
        **overrides: Parâmetros adicionais
    
    Returns:
        Dicionário de configuração completo para treinamento
    """
    return get_config(model_name, data_dir, **overrides)
