"""
Interface de comparação entre modelos e visualização de histórico.
"""

import gradio as gr
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# from experiments.compare import compare_models  # Não existe, vamos usar main() ao invés


class ComparisonManager:
    """Gerencia comparação de modelos e visualização de histórico."""

    def __init__(self):
        self.history_dir = Path("./results/training_history")
        self.results_dir = Path("./results")

    def load_training_histories(self):
        """
        Carrega todos os históricos de treinamento salvos.

        Returns:
            Lista de dicionários com históricos
        """
        if not self.history_dir.exists():
            return []

        histories = []
        for history_file in self.history_dir.glob("*.json"):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    history['filename'] = history_file.name
                    histories.append(history)
            except Exception as e:
                print(f"Erro ao carregar {history_file}: {e}")

        # Ordenar por timestamp (mais recente primeiro)
        histories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return histories

    def get_history_summary(self):
        """
        Retorna resumo dos históricos de treinamento.

        Returns:
            DataFrame com resumo
        """
        histories = self.load_training_histories()

        if not histories:
            return pd.DataFrame(columns=['Modelo', 'Data', 'Épocas', 'Best AUC', 'Test Accuracy'])

        summary_data = []
        for hist in histories:
            try:
                timestamp = hist.get('timestamp', 'N/A')
                if timestamp != 'N/A':
                    date_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    date_str = date_obj.strftime("%d/%m/%Y %H:%M")
                else:
                    date_str = 'N/A'

                model_name = hist.get('model', 'Unknown')
                training_hist = hist.get('training_history', {})
                test_metrics = hist.get('test_metrics', {})

                epochs_trained = len(training_hist.get('train_loss', []))
                best_auc = max(training_hist.get('val_auc', [0]))
                test_accuracy = test_metrics.get('accuracy', 0)

                summary_data.append({
                    'Modelo': model_name,
                    'Data': date_str,
                    'Épocas': epochs_trained,
                    'Best Val AUC': f"{best_auc:.4f}",
                    'Test Accuracy': f"{test_accuracy:.4f}",
                    'Filename': hist['filename']
                })
            except Exception as e:
                print(f"Erro ao processar histórico: {e}")

        return pd.DataFrame(summary_data)

    def plot_training_curves(self, history_filename):
        """
        Plota curvas de treinamento de um histórico específico.

        Args:
            history_filename: Nome do arquivo de histórico

        Returns:
            Figura matplotlib
        """
        history_path = self.history_dir / history_filename

        if not history_path.exists():
            return None

        with open(history_path, 'r') as f:
            history = json.load(f)

        training_hist = history.get('training_history', {})

        if not training_hist:
            return None

        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Curvas de Treinamento - {history.get('model', 'Model')}", fontsize=16, y=0.995)

        epochs = range(1, len(training_hist.get('train_loss', [])) + 1)

        # Loss
        ax = axes[0, 0]
        if 'train_loss' in training_hist and 'val_loss' in training_hist:
            ax.plot(epochs, training_hist['train_loss'], label='Train Loss', marker='o', markersize=3)
            ax.plot(epochs, training_hist['val_loss'], label='Val Loss', marker='s', markersize=3)
            ax.set_xlabel('Época')
            ax.set_ylabel('Loss')
            ax.set_title('Loss de Treinamento e Validação')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # AUC-ROC
        ax = axes[0, 1]
        if 'val_auc' in training_hist:
            ax.plot(epochs, training_hist['val_auc'], label='Val AUC-ROC',
                   marker='o', markersize=3, color='green')
            ax.set_xlabel('Época')
            ax.set_ylabel('AUC-ROC')
            ax.set_title('AUC-ROC na Validação')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.5, 1.0])

        # Learning Rate
        ax = axes[1, 0]
        if 'learning_rate' in training_hist:
            ax.plot(epochs, training_hist['learning_rate'], marker='o', markersize=3, color='red')
            ax.set_xlabel('Época')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Métricas de Teste
        ax = axes[1, 1]
        test_metrics = history.get('test_metrics', {})
        if test_metrics:
            metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'auc_roc']
            values = [test_metrics.get(m, 0) for m in metrics_to_plot]
            labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC']

            bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel('Score')
            ax.set_title('Métricas no Conjunto de Teste')
            ax.set_ylim([0, 1])
            ax.grid(True, axis='y', alpha=0.3)

            # Adicionar valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig

    def compare_saved_models(self):
        """
        Compara os modelos salvos usando os checkpoints.

        Returns:
            Figura com comparação
        """
        resnet_checkpoint = Path("./checkpoints/resnet50/best_model.pth")
        efficientnet_checkpoint = Path("./checkpoints/efficientnet_b0/best_model.pth")

        if not (resnet_checkpoint.exists() and efficientnet_checkpoint.exists()):
            return None, "❌ Um ou mais checkpoints não encontrados. Treine os modelos primeiro."

        # Executar comparação usando o script compare.py
        try:
            import subprocess
            import sys

            # Executar o script de comparação como subprocess
            result = subprocess.run(
                [sys.executable, "experiments/compare.py"],
                capture_output=True,
                text=True,
                cwd=str(Path.cwd())
            )

            if result.returncode == 0:
                # Carregar imagem gerada
                comparison_img = self.results_dir / "metrics_comparison.png"
                if comparison_img.exists():
                    return str(comparison_img), "✅ Comparação concluída!"
                else:
                    return None, "⚠️ Comparação executada, mas imagem não foi gerada."
            else:
                return None, f"❌ Erro ao executar comparação:\n{result.stderr}"

        except Exception as e:
            return None, f"❌ Erro na comparação: {str(e)}"

    def get_best_models_summary(self):
        """
        Retorna resumo dos melhores modelos treinados.

        Returns:
            DataFrame com informações
        """
        histories = self.load_training_histories()

        if not histories:
            return pd.DataFrame()

        # Agrupar por modelo e pegar o melhor de cada
        best_models = {}
        for hist in histories:
            model_name = hist.get('model', 'Unknown')
            test_metrics = hist.get('test_metrics', {})
            auc = test_metrics.get('auc_roc', 0)

            if model_name not in best_models or auc > best_models[model_name].get('auc', 0):
                best_models[model_name] = {
                    'auc': auc,
                    'accuracy': test_metrics.get('accuracy', 0),
                    'sensitivity': test_metrics.get('sensitivity', 0),
                    'specificity': test_metrics.get('specificity', 0),
                    'f1_score': test_metrics.get('f1_score', 0),
                    'timestamp': hist.get('timestamp', 'N/A')
                }

        # Criar DataFrame
        data = []
        for model_name, metrics in best_models.items():
            timestamp = metrics['timestamp']
            if timestamp != 'N/A':
                date_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                date_str = date_obj.strftime("%d/%m/%Y %H:%M")
            else:
                date_str = 'N/A'

            data.append({
                'Modelo': model_name,
                'Data': date_str,
                'AUC-ROC': f"{metrics['auc']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })

        return pd.DataFrame(data)


def create_comparison_interface(comparison_manager):
    """Cria interface para comparação de modelos."""
    gr.Markdown("# ⚖️ Comparação de Modelos")
    gr.Markdown("Compare o desempenho dos modelos ResNet-50 e EfficientNet-B0.")
    
    compare_btn = gr.Button("🔄 Executar Comparação", variant="primary", size="lg")
    status_output = gr.Textbox(label="Status", interactive=False)
    comparison_image = gr.Image(label="Comparação de Métricas", type="filepath")
    
    compare_btn.click(
        fn=comparison_manager.compare_saved_models,
        outputs=[comparison_image, status_output]
    )


def create_history_interface(comparison_manager):
    """Cria interface para visualização de histórico."""
    gr.Markdown("# 📊 Histórico de Treinamentos")
    gr.Markdown("Visualize e analise os históricos de todos os treinamentos realizados.")
    
    refresh_btn = gr.Button("🔄 Atualizar Lista", size="sm")
    history_table = gr.Dataframe(label="Histórico de Treinamentos", interactive=False)
    
    with gr.Row():
        history_select = gr.Dropdown(label="Selecionar Treinamento", choices=[], interactive=True)
        plot_btn = gr.Button("📈 Plotar Curvas", variant="primary")
    
    training_curves_plot = gr.Plot(label="Curvas de Treinamento")
    
    def load_histories():
        summary = comparison_manager.get_history_summary()
        if not summary.empty:
            choices = summary['Filename'].tolist()
            return summary, gr.update(choices=choices)
        return summary, gr.update(choices=[])
    
    def plot_curves(filename):
        if not filename:
            return None
        return comparison_manager.plot_training_curves(filename)
    
    refresh_btn.click(fn=load_histories, outputs=[history_table, history_select])
    plot_btn.click(fn=plot_curves, inputs=history_select, outputs=training_curves_plot)
