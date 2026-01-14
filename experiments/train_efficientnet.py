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
from config.default_config import get_config


def main():
    print("\n" + "="*60)
    print("TREINAMENTO EfficientNet-B0")
    print("="*60 + "\n")
    
    # Validar dataset
    from pathlib import Path
    data_dir = Path('./data/isic2020')
    if not data_dir.exists():
        print(f"❌ Dataset não encontrado: {data_dir}")
        print("   Execute: python data/organize_isic.py --help")
        sys.exit(1)
    
    benign_dir = data_dir / 'benign'
    malignant_dir = data_dir / 'malignant'
    if not benign_dir.exists() or not malignant_dir.exists():
        print(f"❌ Estrutura de dataset inválida em: {data_dir}")
        print("   Esperado: data_dir/benign/ e data_dir/malignant/")
        print("   Execute: python data/organize_isic.py --help")
        sys.exit(1)
    
    # Configuração usando config centralizada
    config = get_config('efficientnet_b0')
    
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
    train_cfg = config['training']
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_cfg['learning_rate'], 
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=train_cfg['scheduler_factor'], 
        patience=train_cfg['scheduler_patience'], 
        min_lr=train_cfg['scheduler_min_lr'],
        verbose=True
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
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'])
    
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

