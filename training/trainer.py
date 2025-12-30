"""
Sistema de treinamento com callbacks (early stopping, checkpointing, logging).
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional


class Trainer:
    """
    Classe para gerenciar treinamento de modelos.
    
    Features:
        - Loop de treino/validação
        - Early stopping
        - Checkpointing automático
        - Logging via TensorBoard
        - Learning rate scheduling
    """
    
    def __init__(self, model, optimizer, criterion, scheduler,
                 device, config: Dict, log_dir: str = './runs',
                 checkpoint_dir: str = './checkpoints'):
        """
        Args:
            model: Modelo PyTorch
            optimizer: Otimizador
            criterion: Função de perda
            scheduler: Learning rate scheduler
            device: Dispositivo (cuda/cpu)
            config: Dicionário de configuração
            log_dir: Diretório para logs TensorBoard
            checkpoint_dir: Diretório para checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Diretórios
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        model_name = config['model']['name']
        self.writer = SummaryWriter(log_dir=self.log_dir / model_name)
        
        # Early stopping
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.patience = config['training']['early_stopping_patience']
        
        # Histórico
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        print(f"\n✓ Trainer inicializado")
        print(f"  Logs: {self.log_dir / model_name}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
    
    def train_epoch(self, train_loader) -> float:
        """
        Executa uma época de treinamento.
        
        Args:
            train_loader: DataLoader de treino
            
        Returns:
            Loss médio da época
        """
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Acumular loss
            running_loss += loss.item() * images.size(0)
            
            # Atualizar progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def validate(self, val_loader) -> tuple:
        """
        Valida modelo.
        
        Args:
            val_loader: DataLoader de validação
            
        Returns:
            Tupla (loss, AUC-ROC)
        """
        from evaluation.metrics import evaluate_model
        
        metrics, val_loss = evaluate_model(
            self.model, val_loader, self.device, self.criterion
        )
        
        return val_loss, metrics['auc_roc']
    
    def fit(self, train_loader, val_loader, epochs: int):
        """
        Treina modelo por múltiplas épocas.
        
        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            epochs: Número de épocas
        """
        print(f"\n{'='*60}")
        print(f"INICIANDO TREINAMENTO")
        print(f"{'='*60}")
        print(f"Épocas: {epochs}")
        print(f"Patience: {self.patience}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            print(f"\nÉpoca {epoch}/{epochs}")
            print("-" * 40)
            
            # Treinar
            train_loss = self.train_epoch(train_loader)
            
            # Validar
            val_loss, val_auc = self.validate(val_loader)
            
            # Learning rate atual
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Atualizar scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_auc)
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/AUC-ROC', val_auc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Imprimir métricas
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val AUC:    {val_auc:.4f}")
            print(f"LR:         {current_lr:.6f}")
            
            # Checkpointing
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ Novo melhor modelo! AUC: {val_auc:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"⚠ Sem melhora por {self.epochs_without_improvement} época(s)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⚠ Early stopping acionado após {epoch} épocas")
                print(f"Melhor AUC: {self.best_val_auc:.4f}")
                break
            
            # Salvar checkpoint periódico
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*60}")
        print(f"TREINAMENTO CONCLUÍDO")
        print(f"Melhor AUC-ROC: {self.best_val_auc:.4f}")
        print(f"{'='*60}\n")
        
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Salva checkpoint do modelo.
        
        Args:
            epoch: Número da época
            is_best: Se é o melhor modelo até agora
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_auc': self.best_val_auc,
            'config': self.config,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"  → Checkpoint salvo: {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Carrega checkpoint.
        
        Args:
            checkpoint_path: Caminho para o checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint.get('history', self.history)
        
        print(f"✓ Checkpoint carregado: {checkpoint_path}")
        print(f"  Época: {checkpoint['epoch']}")
        print(f"  Melhor AUC: {self.best_val_auc:.4f}")
        
        return checkpoint['epoch']