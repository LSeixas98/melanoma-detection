"""
Interface de treinamento com progresso em tempo real.
"""

import gradio as gr
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
from datetime import datetime
from threading import Thread
import time

from models.resnet import ResNet50
from models.efficientnet import EfficientNetB0
from data.dataset import create_data_loaders
from training.trainer import Trainer
from config.default_config import get_training_config
from utils.reproducibility import set_seed, get_device


class TrainingManager:
    """Gerencia o treinamento com atualização de progresso em tempo real."""

    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.training_logs = []
        self.trainer = None

    def start_training(self, model_name, data_dir, epochs, batch_size, learning_rate,
                      early_stopping_patience, progress=gr.Progress()):
        """
        Inicia o treinamento de um modelo.

        Args:
            model_name: 'resnet50' ou 'efficientnet_b0'
            data_dir: Diretório dos dados
            epochs: Número de épocas
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            early_stopping_patience: Paciência para early stopping
            progress: Objeto de progresso do Gradio
        """
        if self.is_training:
            return "⚠️ Já existe um treinamento em andamento!"

        self.is_training = True
        self.current_epoch = 0
        self.total_epochs = epochs
        self.training_logs = []

        try:
            # Configuração
            progress(0, desc="Inicializando configuração...")
            config = get_training_config(
                model_name=model_name,
                data_dir=data_dir,
                training={'epochs': epochs, 'learning_rate': learning_rate,
                         'early_stopping_patience': early_stopping_patience},
                data={'batch_size': batch_size}
            )

            # Seed para reprodutibilidade
            set_seed(config['random_seed'])
            device = get_device()

            # Criar DataLoaders
            progress(0.1, desc="Carregando dados...")
            train_loader, val_loader, test_loader = create_data_loaders(config)

            self.log(f"✓ Dados carregados: {len(train_loader.dataset)} treino, "
                    f"{len(val_loader.dataset)} validação, {len(test_loader.dataset)} teste")

            # Criar modelo
            progress(0.2, desc=f"Criando modelo {model_name}...")
            if model_name == 'resnet50':
                model = ResNet50(num_classes=2, pretrained=True)
            else:
                model = EfficientNetB0(num_classes=2, pretrained=True)

            model = model.to(device)
            self.log(f"✓ Modelo {model_name} criado e movido para {device}")

            # Otimizador e scheduler
            optimizer = Adam(model.parameters(), lr=learning_rate,
                           weight_decay=config['training']['weight_decay'])

            scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=config['training']['scheduler_factor'],
                patience=config['training']['scheduler_patience'],
                min_lr=config['training']['scheduler_min_lr']
            )

            # Loss com pesos de classe
            class_weights = torch.tensor(config['loss']['class_weights'],
                                        dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # Criar Trainer customizado
            progress(0.3, desc="Inicializando trainer...")
            trainer = CustomTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                config=config,
                log_dir=config['log_dir'],
                checkpoint_dir=config['checkpoint_dir'],
                training_manager=self
            )

            self.trainer = trainer
            self.log(f"✓ Trainer inicializado. Iniciando treinamento por {epochs} épocas...")

            # Treinar
            trainer.fit(train_loader, val_loader, epochs)

            # Avaliar no teste
            from evaluation.metrics import evaluate_model
            progress(0.95, desc="Avaliando no conjunto de teste...")

            metrics, test_loss = evaluate_model(model, test_loader, device, criterion)

            self.log("\n" + "="*60)
            self.log("RESULTADOS NO CONJUNTO DE TESTE")
            self.log("="*60)
            self.log(f"Loss: {test_loss:.4f}")
            self.log(f"Acurácia: {metrics['accuracy']:.4f}")
            self.log(f"Sensibilidade: {metrics['sensitivity']:.4f}")
            self.log(f"Especificidade: {metrics['specificity']:.4f}")
            self.log(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            self.log("="*60)

            # Salvar histórico
            self._save_training_history(config, metrics, test_loss)

            self.is_training = False
            return "✅ Treinamento concluído com sucesso!"

        except Exception as e:
            self.is_training = False
            error_msg = f"❌ Erro durante o treinamento: {str(e)}"
            self.log(error_msg)
            return error_msg

    def log(self, message):
        """Adiciona mensagem ao log de treinamento."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.training_logs.append(log_entry)
        print(log_entry)

    def get_logs(self):
        """Retorna os logs de treinamento."""
        return "\n".join(self.training_logs)

    def get_progress(self):
        """Retorna o progresso atual do treinamento."""
        if not self.is_training:
            return 0, "Nenhum treinamento em andamento"

        progress_pct = (self.current_epoch / self.total_epochs) * 100
        return progress_pct, f"Época {self.current_epoch}/{self.total_epochs}"

    def _save_training_history(self, config, test_metrics, test_loss):
        """Salva histórico de treinamento."""
        history_dir = Path("./results/training_history")
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config['model']['name']

        history = {
            'model': model_name,
            'timestamp': timestamp,
            'config': config,
            'test_metrics': test_metrics,
            'test_loss': test_loss,
            'training_history': self.trainer.history if self.trainer else {}
        }

        filename = history_dir / f"{model_name}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)

        self.log(f"✓ Histórico salvo em: {filename}")


class CustomTrainer(Trainer):
    """Trainer customizado que atualiza o TrainingManager."""

    def __init__(self, *args, training_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_manager = training_manager

    def fit(self, train_loader, val_loader, epochs: int):
        """Sobrescreve fit para atualizar progresso."""
        print(f"\n{'='*60}")
        print(f"INICIANDO TREINAMENTO")
        print(f"{'='*60}")
        print(f"Épocas: {epochs}")
        print(f"Patience: {self.patience}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            if self.training_manager:
                self.training_manager.current_epoch = epoch
                self.training_manager.log(f"\n{'='*40}")
                self.training_manager.log(f"Época {epoch}/{epochs}")
                self.training_manager.log("="*40)

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

            # Log para o manager
            if self.training_manager:
                self.training_manager.log(f"Train Loss: {train_loss:.4f}")
                self.training_manager.log(f"Val Loss:   {val_loss:.4f}")
                self.training_manager.log(f"Val AUC:    {val_auc:.4f}")
                self.training_manager.log(f"LR:         {current_lr:.6f}")

            # Checkpointing
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                if self.training_manager:
                    self.training_manager.log(f"✓ Novo melhor modelo! AUC: {val_auc:.4f}")
            else:
                self.epochs_without_improvement += 1
                if self.training_manager:
                    self.training_manager.log(
                        f"⚠ Sem melhora por {self.epochs_without_improvement} época(s)"
                    )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                if self.training_manager:
                    self.training_manager.log(f"\n⚠ Early stopping acionado após {epoch} épocas")
                    self.training_manager.log(f"Melhor AUC: {self.best_val_auc:.4f}")
                break

            # Salvar checkpoint periódico
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        if self.training_manager:
            self.training_manager.log(f"\n{'='*60}")
            self.training_manager.log(f"TREINAMENTO CONCLUÍDO")
            self.training_manager.log(f"Melhor AUC-ROC: {self.best_val_auc:.4f}")
            self.training_manager.log(f"{'='*60}\n")

        self.writer.close()


def create_training_interface(training_manager):
    """
    Cria interface Gradio para treinamento.
    
    Args:
        training_manager: Instância de TrainingManager
        
    Returns:
        Componente Gradio
    """
    gr.Markdown("# 🏋️ Treinamento de Modelos")
    gr.Markdown("Configure e treine os modelos ResNet-50 ou EfficientNet-B0 para detecção de melanoma.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configurações")
            
            model_choice = gr.Dropdown(
                choices=['resnet50', 'efficientnet_b0'],
                value='resnet50',
                label="Modelo",
                info="Escolha a arquitetura da CNN"
            )
            
            data_dir = gr.Textbox(
                value="./data/isic2020",
                label="Diretório dos Dados",
                info="Caminho para a pasta com as imagens"
            )
            
            epochs = gr.Slider(
                minimum=1, maximum=100, value=50, step=1,
                label="Épocas",
                info="Número máximo de épocas de treinamento"
            )
            
            batch_size = gr.Slider(
                minimum=8, maximum=128, value=32, step=8,
                label="Batch Size",
                info="Tamanho do lote para treinamento"
            )
            
            learning_rate = gr.Number(
                value=0.0001,
                label="Learning Rate",
                info="Taxa de aprendizado inicial"
            )
            
            patience = gr.Slider(
                minimum=3, maximum=20, value=10, step=1,
                label="Early Stopping Patience",
                info="Épocas sem melhora antes de parar"
            )
            
            train_btn = gr.Button("▶️ Iniciar Treinamento", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### Progresso e Logs")
            
            status_output = gr.Textbox(
                label="Status",
                value="Aguardando início do treinamento...",
                interactive=False
            )
            
            logs_output = gr.Textbox(
                label="Logs de Treinamento",
                lines=20,
                max_lines=30,
                interactive=False,
                autoscroll=True
            )
            
            refresh_btn = gr.Button("🔄 Atualizar Logs", size="sm")
    
    # Eventos
    def train_wrapper(*args):
        return training_manager.start_training(*args)
    
    def refresh_logs():
        return training_manager.get_logs()
    
    train_btn.click(
        fn=train_wrapper,
        inputs=[model_choice, data_dir, epochs, batch_size, learning_rate, patience],
        outputs=status_output
    )
    
    refresh_btn.click(
        fn=refresh_logs,
        outputs=logs_output
    )
