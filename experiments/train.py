"""
Script principal para treinamento de modelos.

Uso:
    python experiments/train.py --config config/resnet50_config.yaml
    python experiments/train.py --config config/efficientnet_config.yaml
"""

import sys
sys.path.append('.')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Importar módulos do projeto
from utils.reproducibility import set_seed, get_device
from utils.config import load_config, DEFAULT_CONFIG
from data.preprocessing import get_transforms
from data.dataset import get_dataloaders
from models.model_factory import create_model, get_model_info
from training.trainer import Trainer


def setup_optimizer(model, config):
    """
    Configura otimizador baseado em config.
    
    Args:
        model: Modelo PyTorch
        config: Dicionário de configuração
        
    Returns:
        Otimizador configurado
    """
    opt_config = config['training']
    
    if opt_config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay']
        )
    elif opt_config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config['learning_rate'],
            momentum=0.9,
            weight_decay=opt_config['weight_decay']
        )
    else:
        raise ValueError(f"Otimizador não reconhecido: {opt_config['optimizer']}")
    
    print(f"✓ Otimizador: {opt_config['optimizer'].upper()}")
    print(f"  LR: {opt_config['learning_rate']}")
    print(f"  Weight decay: {opt_config['weight_decay']}")
    
    return optimizer


def setup_scheduler(optimizer, config):
    """
    Configura learning rate scheduler.
    
    Args:
        optimizer: Otimizador PyTorch
        config: Dicionário de configuração
        
    Returns:
        Scheduler ou None
    """
    sched_config = config['training']
    
    if sched_config['scheduler'].lower() == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximizar AUC
            factor=sched_config['scheduler_factor'],
            patience=sched_config['scheduler_patience'],
            min_lr=sched_config['scheduler_min_lr'],
            verbose=True
        )
        print(f"✓ Scheduler: ReduceLROnPlateau")
        print(f"  Patience: {sched_config['scheduler_patience']}")
        print(f"  Factor: {sched_config['scheduler_factor']}")
    else:
        scheduler = None
        print("⚠ Nenhum scheduler configurado")
    
    return scheduler


def setup_criterion(config, class_weights, device):
    """
    Configura função de perda.
    
    Args:
        config: Dicionário de configuração
        class_weights: Pesos das classes
        device: Dispositivo
        
    Returns:
        Função de perda
    """
    loss_config = config['loss']
    
    if loss_config['type'].lower() == 'weighted_cross_entropy':
        # Usar pesos da config ou calculados
        if 'class_weights' in loss_config:
            weights = torch.tensor(loss_config['class_weights'], dtype=torch.float32)
        else:
            weights = class_weights
        
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        print(f"✓ Loss: Weighted Cross-Entropy")
        print(f"  Pesos: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"✓ Loss: Cross-Entropy (sem pesos)")
    
    return criterion


def main(args):
    """
    Função principal de treinamento.
    """
    print("\n" + "="*60)
    print("TREINAMENTO DE MODELO PARA DETECÇÃO DE MELANOMA")
    print("="*60 + "\n")
    
    # Carregar configuração
    if args.config:
        config = load_config(args.config)
    else:
        print("⚠ Nenhuma config fornecida, usando DEFAULT_CONFIG")
        config = DEFAULT_CONFIG
    
    # Reprodutibilidade
    set_seed(config['random_seed'])
    
    # Device
    device = get_device()
    
    # Transformações
    print("\n[1/7] Preparando transformações...")
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    
    # DataLoaders
    print("\n[2/7] Carregando dataset...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        config, train_transform, val_transform
    )
    
    # Modelo
    print("\n[3/7] Criando modelo...")
    model = create_model(config)
    model = model.to(device)
    
    model_info = get_model_info(model)
    print(f"  Nome: {model_info['name']}")
    print(f"  Parâmetros: {model_info['trainable_params']:,}")
    print(f"  Tamanho: {model_info['size_mb']:.2f} MB")
    
    # Otimizador
    print("\n[4/7] Configurando otimizador...")
    optimizer = setup_optimizer(model, config)
    
    # Scheduler
    print("\n[5/7] Configurando scheduler...")
    scheduler = setup_scheduler(optimizer, config)
    
    # Loss function
    print("\n[6/7] Configurando função de perda...")
    criterion = setup_criterion(config, class_weights, device)
    
    # Trainer
    print("\n[7/7] Inicializando Trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config=config,
        log_dir=config.get('log_dir', './runs'),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints')
    )
    
    # Treinar
    epochs = config['training']['epochs']
    trainer.fit(train_loader, val_loader, epochs)
    
    # Avaliar no conjunto de teste
    print("\n" + "="*60)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*60)
    
    from evaluation.metrics import evaluate_model, print_metrics
    
    # Carregar melhor modelo
    best_model_path = Path(config.get('checkpoint_dir', './checkpoints')) / 'best_model.pth'
    if best_model_path.exists():
        print(f"\nCarregando melhor modelo: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_loss = evaluate_model(model, test_loader, device, criterion)
    
    print_metrics(test_metrics, phase="Test")
    
    print("✓ Treinamento e avaliação concluídos!")
    print(f"  Melhor modelo salvo em: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar modelo para detecção de melanoma')
    parser.add_argument('--config', type=str, default=None,
                       help='Caminho para arquivo de configuração YAML')
    
    args = parser.parse_args()
    main(args)