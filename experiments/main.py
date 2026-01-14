"""
Script principal para executar todo o pipeline em sequência:
1. Treinar ResNet-50
2. Treinar EfficientNet-B0
3. Comparar modelos
4. Coletar resultados para relatório

Uso:
    python experiments/run_all.py
"""

import sys
sys.path.append('.')

import subprocess
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from utils.reproducibility import set_seed, get_device
from data.preprocessing import get_transforms
from data.dataset import get_dataloaders
from models.resnet import get_resnet50
from models.efficientnet import get_efficientnet_b0
from evaluation.metrics import evaluate_model, get_predictions
from evaluation.efficiency import benchmark_model
from training.trainer import Trainer
import torch.nn as nn
import torch.optim as optim


class ResultsCollector:
    """Coleta e armazena resultados de treinamento e comparação."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'resnet50': {},
            'efficientnet_b0': {},
            'comparison': {}
        }
    
    def save(self, path='./results/results.json'):
        """Salva resultados em JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ Resultados salvos em: {path}")


def train_resnet(collector):
    """Treina ResNet-50 e coleta resultados."""
    print("\n" + "="*80)
    print("ETAPA 1/3: TREINAMENTO ResNet-50")
    print("="*80 + "\n")
    
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
        'checkpoint_dir': './checkpoints/resnet50',
        'log_dir': './runs',
        'training': {
            'epochs': 50,
            'early_stopping_patience': 10
        }
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
    print("\n[2/5] Criando ResNet-50...")
    model = get_resnet50(num_classes=2, pretrained=True).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros: {num_params:,}")
    
    # Treinamento
    print("\n[3/5] Configurando treinamento...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
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
    
    print("\n[4/5] Treinando...")
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'])
    
    # Avaliar e coletar resultados
    print("\n[5/5] Avaliando no conjunto de teste...")
    best_model_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_loss = evaluate_model(model, test_loader, device, criterion)
    
    # Eficiência
    efficiency = benchmark_model(model, device, verbose=False)
    
    # Salvar resultados
    collector.results['resnet50'] = {
        'test_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) 
                         for k, v in test_metrics.items() if k != 'confusion_matrix'},
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'test_loss': float(test_loss) if test_loss else None,
        'best_val_auc': float(trainer.best_val_auc),
        'num_parameters': int(num_params),
        'training_history': {
            'train_loss': [float(x) for x in trainer.history['train_loss']],
            'val_loss': [float(x) for x in trainer.history['val_loss']],
            'val_auc': [float(x) for x in trainer.history['val_auc']]
        },
        'efficiency': efficiency if efficiency else {}
    }
    
    print(f"\n✓ ResNet-50 treinado e avaliado!")
    print(f"  Melhor AUC-ROC: {trainer.best_val_auc:.4f}")
    print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    return model, test_loader, device


def train_efficientnet(collector):
    """Treina EfficientNet-B0 e coleta resultados."""
    print("\n" + "="*80)
    print("ETAPA 2/3: TREINAMENTO EfficientNet-B0")
    print("="*80 + "\n")
    
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
        'log_dir': './runs',
        'training': {
            'epochs': 50,
            'early_stopping_patience': 10
        }
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
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros: {num_params:,}")
    
    # Treinamento
    print("\n[3/5] Configurando treinamento...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
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
    
    print("\n[4/5] Treinando...")
    trainer.fit(train_loader, val_loader, epochs=config['training']['epochs'])
    
    # Avaliar e coletar resultados
    print("\n[5/5] Avaliando no conjunto de teste...")
    best_model_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_loss = evaluate_model(model, test_loader, device, criterion)
    
    # Eficiência
    efficiency = benchmark_model(model, device, verbose=False)
    
    # Salvar resultados
    collector.results['efficientnet_b0'] = {
        'test_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) 
                         for k, v in test_metrics.items() if k != 'confusion_matrix'},
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'test_loss': float(test_loss) if test_loss else None,
        'best_val_auc': float(trainer.best_val_auc),
        'num_parameters': int(num_params),
        'training_history': {
            'train_loss': [float(x) for x in trainer.history['train_loss']],
            'val_loss': [float(x) for x in trainer.history['val_loss']],
            'val_auc': [float(x) for x in trainer.history['val_auc']]
        },
        'efficiency': efficiency if efficiency else {}
    }
    
    print(f"\n✓ EfficientNet-B0 treinado e avaliado!")
    print(f"  Melhor AUC-ROC: {trainer.best_val_auc:.4f}")
    print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    return model, test_loader, device


def compare_models(collector):
    """Compara modelos e coleta resultados."""
    print("\n" + "="*80)
    print("ETAPA 3/3: COMPARAÇÃO DE MODELOS")
    print("="*80 + "\n")
    
    # Executar script de comparação
    print("Executando comparação completa...")
    result = subprocess.run(
        [sys.executable, 'experiments/compare.py'],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("⚠ Erro ao executar comparação")
    
    # Carregar resultados da comparação (se salvos)
    comparison_file = Path('./results/comparison_results.json')
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            collector.results['comparison'] = json.load(f)
    
    print("\n✓ Comparação concluída!")


def main():
    """Função principal."""
    print("\n" + "="*80)
    print("PIPELINE COMPLETO: Treinamento e Comparação")
    print("="*80)
    print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    collector = ResultsCollector()
    
    try:
        # Etapa 1: Treinar ResNet-50
        train_resnet(collector)
        
        # Etapa 2: Treinar EfficientNet-B0
        train_efficientnet(collector)
        
        # Etapa 3: Comparar modelos
        compare_models(collector)
        
        # Salvar resultados
        collector.save('./results/results.json')
        
        print("\n" + "="*80)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print(f"\nResultados salvos em: ./results/results.json")
        print("Execute 'python experiments/generate_report.py' para gerar relatório completo.")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrompido pelo usuário")
        collector.save('./results/results_partial.json')
    except Exception as e:
        print(f"\n\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        collector.save('./results/results_error.json')
        raise


if __name__ == '__main__':
    main()

