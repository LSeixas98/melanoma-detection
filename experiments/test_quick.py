"""
Script rápido para testar o pipeline com dataset menor (isic2020_test).
Use este script para verificar se tudo está funcionando antes do treinamento completo.

Uso:
    python experiments/test_quick.py
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
from models.resnet import get_resnet50
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, print_metrics
from config.default_config import get_config


def main():
    print("\n" + "="*60)
    print("TESTE RÁPIDO - ResNet-50 com dataset de teste")
    print("="*60 + "\n")
    
    # Validar dataset de teste
    data_dir = Path('./data/isic2020_test')
    if not data_dir.exists():
        print(f"❌ Dataset de teste não encontrado: {data_dir}")
        print("   Execute: python data/create_subset.py --source ./data/isic2020 --target ./data/isic2020_test --size 0.05")
        sys.exit(1)
    
    benign_dir = data_dir / 'benign'
    malignant_dir = data_dir / 'malignant'
    if not benign_dir.exists() or not malignant_dir.exists():
        print(f"❌ Estrutura de dataset inválida em: {data_dir}")
        print("   Esperado: data_dir/benign/ e data_dir/malignant/")
        sys.exit(1)
    
    # Configuração para teste com dataset de 20%
    config = get_config('resnet50', data_dir='./data/isic2020_test')
    
    # Configuração otimizada para validação com 20% do dataset
    config['training']['epochs'] = 15  # 15 épocas para validação mais precisa
    config['data']['batch_size'] = 32  # Batch size padrão
    config['checkpoint_dir'] = './checkpoints/resnet50_test'
    
    set_seed(config['random_seed'])
    device = get_device()
    
    print(f"📊 Configuração de validação (20% do dataset):")
    print(f"   Dataset: {config['data']['data_dir']}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Épocas: {config['training']['epochs']}")
    print(f"   Device: {device}\n")
    
    # Dados
    print("[1/4] Carregando dados...")
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        config, train_transform, val_transform
    )
    
    print(f"   ✓ Train: {len(train_loader.dataset)} imagens")
    print(f"   ✓ Val: {len(val_loader.dataset)} imagens")
    print(f"   ✓ Test: {len(test_loader.dataset)} imagens")
    
    # Modelo
    print("\n[2/4] Criando ResNet-50...")
    model = get_resnet50(num_classes=2, pretrained=True).to(device)
    print(f"  ✓ Modelo criado: {sum(p.numel() for p in model.parameters()):,} parâmetros")
    
    # Otimizador e Loss
    print("\n[3/4] Configurando treinamento...")
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
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config={'model': {'name': 'resnet50'}, **config},
        log_dir=config['log_dir'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Treinar
    print("\n[4/4] Treinando (validação com 20% do dataset - 15 épocas)...")
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'])
    
    # Avaliar
    print("\n" + "="*60)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*60)
    
    best_model_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Melhor modelo carregado")
    
    test_metrics, _ = evaluate_model(model, test_loader, device, criterion)
    print_metrics(test_metrics, phase="Test")
    
    print("\n" + "="*60)
    print("✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print("\nResultados obtidos com 20% do dataset e 15 épocas.")
    print("Se os resultados estão satisfatórios, você pode executar:")
    print("  python experiments/train_resnet.py  (dataset completo)")
    print("  python experiments/train_efficientnet.py  (dataset completo)")
    print("  python experiments/main.py  (pipeline completo)")


if __name__ == '__main__':
    main()
