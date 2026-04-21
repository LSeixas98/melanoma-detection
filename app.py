"""
Aplicação principal - Interface gráfica completa para detecção de melanoma.

Este aplicativo Gradio fornece uma interface unificada para:
- Treinamento de modelos (ResNet-50 e EfficientNet-B0)
- Predição em imagens únicas e em lote
- Visualização de explicabilidade (Grad-CAM)
- Comparação de modelos
- Análise de histórico de treinamento

Uso:
    python app.py

    O aplicativo será iniciado em http://localhost:7860
"""

import gradio as gr
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from gui.training_interface import TrainingManager, create_training_interface
from gui.prediction_interface import PredictionManager, create_prediction_interface, create_batch_prediction_interface
from gui.comparison_interface import ComparisonManager, create_comparison_interface, create_history_interface


def create_home_interface():
    """Cria a interface inicial/home."""
    gr.Markdown(
            """
            # 🏥 Sistema de Detecção de Melanoma

            ## Bem-vindo ao Sistema de Análise de Lesões Cutâneas com Deep Learning

            Este sistema utiliza redes neurais convolucionais (CNNs) para auxiliar na detecção
            precoce de melanoma através da análise de imagens de lesões cutâneas.

            ### 🎯 Funcionalidades Principais

            #### 🏋️ Treinamento
            - Treine modelos ResNet-50 ou EfficientNet-B0
            - Acompanhe o progresso em tempo real
            - Configuração flexível de hiperparâmetros
            - Early stopping automático
            - Salvamento de checkpoints

            #### 🔍 Predição
            - Análise de imagens individuais
            - Processamento em lote
            - Visualização de probabilidades
            - Comparação entre modelos

            #### 🔬 Explicabilidade
            - Visualização Grad-CAM
            - Identificação de regiões importantes
            - Interpretação visual das decisões

            #### ⚖️ Comparação
            - Compare ResNet-50 vs EfficientNet-B0
            - Análise de métricas clínicas
            - Gráficos comparativos

            #### 📊 Histórico
            - Visualize todos os treinamentos realizados
            - Curvas de aprendizado
            - Evolução de métricas

            ### 🚀 Como Usar

            1. **Preparar Dados**: Organize suas imagens no formato esperado em `./data/isic2020/`
               - `benign/` - Imagens de lesões benignas
               - `malignant/` - Imagens de lesões malignas

            2. **Treinar Modelo**: Vá para a aba "Treinamento" e configure os parâmetros

            3. **Fazer Predições**: Use a aba "Predição" para analisar novas imagens

            4. **Analisar Resultados**: Explore "Comparação" e "Histórico" para insights

            ### 📋 Modelos Disponíveis

            | Modelo | Parâmetros | Características |
            |--------|-----------|-----------------|
            | **ResNet-50** | ~25M | Alta capacidade, mais robusto |
            | **EfficientNet-B0** | ~5.3M | Mais eficiente, menor latência |

            ### ⚠️ Importante

            Este sistema é destinado para **fins educacionais e de pesquisa**.
            Não substitui o diagnóstico médico profissional.

            ### 📚 Dataset

            O sistema foi desenvolvido para trabalhar com o **ISIC 2020 Challenge Dataset**
            (International Skin Imaging Collaboration).

            ---

            **Desenvolvido com:** PyTorch, Gradio, Albumentations, Grad-CAM
            """
    )


def create_about_interface():
    """Cria a interface de informações sobre o sistema."""
    gr.Markdown(
            """
            # ℹ️ Sobre o Sistema

            ## Arquitetura do Projeto

            ### Estrutura de Diretórios

            ```
            melanoma-detection/
            ├── config/              # Configurações centralizadas
            ├── data/                # Datasets e preprocessamento
            ├── models/              # Arquiteturas de modelos
            ├── training/            # Sistema de treinamento
            ├── evaluation/          # Métricas e benchmarking
            ├── explainability/      # Grad-CAM e interpretabilidade
            ├── experiments/         # Scripts de experimentos
            ├── gui/                 # Interface gráfica (este app)
            ├── checkpoints/         # Modelos salvos
            ├── results/             # Resultados e visualizações
            └── runs/                # Logs TensorBoard
            ```

            ### Tecnologias Utilizadas

            #### Deep Learning
            - **PyTorch 2.0+**: Framework principal
            - **torchvision**: Modelos pré-treinados
            - **Albumentations**: Augmentação de dados

            #### Interface & Visualização
            - **Gradio**: Interface web interativa
            - **Matplotlib / Seaborn**: Visualizações
            - **Plotly**: Gráficos interativos
            - **TensorBoard**: Monitoramento de treinamento

            #### Avaliação
            - **scikit-learn**: Métricas clínicas
            - **Grad-CAM**: Explicabilidade visual

            ### Métricas Clínicas

            O sistema calcula as seguintes métricas:

            - **Acurácia**: Proporção de predições corretas
            - **Sensibilidade (Recall)**: Taxa de verdadeiros positivos (detecção de malignos)
            - **Especificidade**: Taxa de verdadeiros negativos (detecção de benignos)
            - **Precisão**: Proporção de positivos corretos
            - **F1-Score**: Média harmônica entre precisão e recall
            - **AUC-ROC**: Área sob a curva ROC (discriminação)

            ### Pipeline de Treinamento

            1. **Carregamento de Dados**
               - Divisão estratificada (70% treino, 15% validação, 15% teste)
               - Weighted sampling para balanceamento

            2. **Augmentação**
               - Rotação (±30°)
               - Flips horizontal/vertical
               - Ajustes de brilho/contraste
               - Shift-Scale-Rotate

            3. **Treinamento**
               - Otimizador: Adam (LR=0.0001)
               - Loss: Weighted Cross-Entropy
               - Scheduler: ReduceLROnPlateau
               - Early Stopping (patience=10)

            4. **Avaliação**
               - Métricas clínicas completas
               - Curva ROC
               - Matriz de confusão
               - Benchmarking de eficiência

            ### Grad-CAM (Explicabilidade)

            O Gradient-weighted Class Activation Mapping (Grad-CAM) permite visualizar
            quais regiões da imagem foram importantes para a decisão do modelo.

            - **Camada alvo ResNet-50**: layer4 (última conv)
            - **Camada alvo EfficientNet**: features[-1] (último bloco)

            ### Configuração Padrão

            ```python
            batch_size = 32
            learning_rate = 0.0001
            max_epochs = 50
            early_stopping_patience = 10
            image_size = 224x224
            optimizer = Adam
            weight_decay = 1e-5
            ```

            ### Hardware Recomendado

            - **GPU**: NVIDIA com CUDA (mínimo 6GB VRAM)
            - **RAM**: 16GB+
            - **Armazenamento**: 10GB+ para datasets

            O sistema detecta automaticamente GPU disponível e usa CPU como fallback.

            ### Referências

            - He et al. (2016) - Deep Residual Learning for Image Recognition (ResNet)
            - Tan & Le (2019) - EfficientNet: Rethinking Model Scaling for CNNs
            - Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks
            - ISIC 2020 Challenge - International Skin Imaging Collaboration

            ### Contato & Contribuições

            Este é um projeto open-source para fins educacionais e de pesquisa.

            ---

            **Versão**: 1.0.0
            **Última atualização**: 2026
            """
    )


def main():
    """Função principal que cria e lança a aplicação."""

    print("\n" + "="*70)
    print("🏥 SISTEMA DE DETECÇÃO DE MELANOMA")
    print("="*70)
    print("Inicializando interface gráfica...")
    print("="*70 + "\n")

    # Criar diretórios necessários
    Path("./checkpoints").mkdir(exist_ok=True)
    Path("./results").mkdir(exist_ok=True)
    Path("./runs").mkdir(exist_ok=True)
    Path("./results/training_history").mkdir(parents=True, exist_ok=True)

    # Instanciar managers
    print("📦 Inicializando managers...")
    training_manager = TrainingManager()
    prediction_manager = PredictionManager()
    comparison_manager = ComparisonManager()
    print("✓ Managers inicializados\n")

    # Criar aplicação com tabs
    print("🎨 Criando aplicação...")
    with gr.Blocks(title="Sistema de Detecção de Melanoma") as app:

        # Título principal
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <h1 style="margin-bottom: 10px;">🏥 Sistema de Detecção de Melanoma</h1>
                <p style="color: #666; font-size: 1.1em;">
                    Análise de Lesões Cutâneas com Deep Learning
                </p>
            </div>
            """
        )

        # Tabs
        with gr.Tabs():
            with gr.Tab("🏠 Início"):
                create_home_interface()

            with gr.Tab("🏋️ Treinamento"):
                create_training_interface(training_manager)

            with gr.Tab("🔍 Predição Individual"):
                create_prediction_interface(prediction_manager)

            with gr.Tab("📁 Predição em Lote"):
                create_batch_prediction_interface(prediction_manager)

            with gr.Tab("⚖️ Comparação"):
                create_comparison_interface(comparison_manager)

            with gr.Tab("📊 Histórico"):
                create_history_interface(comparison_manager)

            with gr.Tab("ℹ️ Sobre"):
                create_about_interface()

        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666; font-size: 0.9em; padding: 10px;">
                Desenvolvido com PyTorch, Gradio & Grad-CAM |
                Para fins educacionais e de pesquisa
            </div>
            """
        )

    print("✓ Aplicação montada\n")
    print("="*70)
    print("🌐 Iniciando servidor Gradio...")
    print("="*70)
    print("\n📍 Acesse a interface em: http://localhost:7860")
    print("\n💡 Dica: Use Ctrl+C para encerrar o servidor\n")

    # Lançar aplicação
    app.launch(
        server_name="0.0.0.0",  # Permite acesso externo
        server_port=7860,
        share=False,  # Defina como True para gerar link público temporário
        show_error=True
    )


if __name__ == "__main__":
    main()
