"""
Script simples para treinar EfficientNet-B0.

Uso:
    python experiments/train_efficientnet.py
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from utils.reproducibility import set_seed, get_device
from data.preprocessing import get_transforms
from data.dataset import get_dataloaders
from models.efficientnet import get_efficientnet_b0
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, print_metrics


def main():
    print("\n" + "="*60)
    print("TREINAMENTO EfficientNet-B0")
    print("="*60 + "\n")
    
    # Configuração simples
    config = {
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
        'random_seed': 42,
        'checkpoint_dir': './checkpoints/efficientnet_b0',
        'log_dir': './runs'
    }
    
    set_seed(config['random_seed'])
    device = get_device()
    
    # Dados
    print("[1/5] Carregando dados...")
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        config, train_transform, val_transform
    )
    
    # Modelo
    print("\n[2/5] Criando EfficientNet-B0...")
    model = get_efficientnet_b0(num_classes=2, pretrained=True).to(device)
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Otimizador e Loss
    print("\n[3/5] Configurando treinamento...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Trainer
    print("\n[4/5] Inicializando Trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config={'model': {'name': 'efficientnet_b0'}, **config},
        log_dir=config['log_dir'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Treinar
    print("\n[5/5] Treinando...")
    trainer.fit(train_loader, val_loader, epochs=50)
    
    # Avaliar
    print("\n" + "="*60)
    print("AVALIAÇÃO NO TESTE")
    print("="*60)
    
    best_model_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, _ = evaluate_model(model, test_loader, device, criterion)
    print_metrics(test_metrics, phase="Test")
    
    print(f"\n✓ Modelo salvo em: {best_model_path}")


if __name__ == '__main__':
    main()

