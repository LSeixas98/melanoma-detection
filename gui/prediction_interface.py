"""
Interface de predição com visualização de Grad-CAM.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from models.resnet import ResNet50
from models.efficientnet import EfficientNetB0
from data.preprocessing import get_transforms
from explainability.gradcam import GradCAM
from utils.reproducibility import get_device


class PredictionManager:
    """Gerencia predições e visualizações Grad-CAM."""

    def __init__(self):
        self.device = get_device()
        self.models = {}
        self.class_names = ['Benigno', 'Maligno']

    def load_model(self, model_name, checkpoint_path):
        """
        Carrega um modelo treinado.

        Args:
            model_name: 'resnet50' ou 'efficientnet_b0'
            checkpoint_path: Caminho para o checkpoint

        Returns:
            Mensagem de status
        """
        try:
            if not Path(checkpoint_path).exists():
                return f"❌ Checkpoint não encontrado: {checkpoint_path}"

            # Criar modelo
            if model_name == 'resnet50':
                model = ResNet50(num_classes=2, pretrained=False)
                target_layer = model.backbone.layer4  # Última camada convolucional
            else:
                model = EfficientNetB0(num_classes=2, pretrained=False)
                target_layer = model.backbone.features[-1]

            # Carregar pesos
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            # Salvar modelo e target layer
            self.models[model_name] = {
                'model': model,
                'target_layer': target_layer,
                'checkpoint': checkpoint
            }

            best_auc = checkpoint.get('best_val_auc', 'N/A')
            return f"✅ Modelo {model_name} carregado com sucesso! (Best AUC: {best_auc:.4f})"

        except Exception as e:
            return f"❌ Erro ao carregar modelo: {str(e)}"

    def predict_single(self, image, model_name, show_gradcam=True):
        """
        Faz predição em uma imagem única.

        Args:
            image: Imagem PIL ou numpy array
            model_name: Nome do modelo a usar
            show_gradcam: Se deve gerar visualização Grad-CAM

        Returns:
            Tupla (predição, probabilidades, imagem_gradcam)
        """
        if model_name not in self.models:
            return "❌ Modelo não carregado!", None, None

        try:
            # Converter para PIL se necessário
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Salvar imagem original para Grad-CAM
            original_image = np.array(image.convert('RGB'))

            # Transformações
            _, val_transform = get_transforms(image_size=224)
            input_tensor = val_transform(image=original_image)['image'].unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Predição
            model_data = self.models[model_name]
            model = model_data['model']

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()

            # Criar texto de resultado
            pred_text = f"**Predição: {self.class_names[predicted_class]}**\n\n"
            pred_text += f"**Probabilidades:**\n"
            pred_text += f"- Benigno: {probabilities[0, 0].item():.2%}\n"
            pred_text += f"- Maligno: {probabilities[0, 1].item():.2%}"

            # Criar gráfico de probabilidades
            probs_dict = {
                'Benigno': probabilities[0, 0].item(),
                'Maligno': probabilities[0, 1].item()
            }

            # Grad-CAM
            gradcam_image = None
            if show_gradcam:
                target_layer = model_data['target_layer']
                gradcam = GradCAM(model, target_layer)

                # Gerar CAM
                cam = gradcam.generate_cam(input_tensor, target_class=predicted_class)

                # Visualizar
                gradcam_image = gradcam.visualize(original_image, cam, alpha=0.5)

            return pred_text, probs_dict, gradcam_image

        except Exception as e:
            return f"❌ Erro na predição: {str(e)}", None, None

    def compare_models(self, image, show_gradcam=True):
        """
        Compara predições de ambos os modelos.

        Args:
            image: Imagem PIL ou numpy array
            show_gradcam: Se deve gerar visualizações Grad-CAM

        Returns:
            Dicionário com resultados de ambos os modelos
        """
        if len(self.models) < 2:
            return {
                'error': "❌ Carregue ambos os modelos para comparação!"
            }

        results = {}

        for model_name in ['resnet50', 'efficientnet_b0']:
            if model_name in self.models:
                pred_text, probs, gradcam_img = self.predict_single(
                    image, model_name, show_gradcam
                )
                results[model_name] = {
                    'prediction': pred_text,
                    'probabilities': probs,
                    'gradcam': gradcam_img
                }

        return results


def create_prediction_interface(prediction_manager):
    """Cria interface Gradio para predição."""
    gr.Markdown("# 🔍 Predição de Melanoma")
    gr.Markdown("Carregue uma imagem de lesão cutânea para análise usando os modelos treinados.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuração")
            
            model_select = gr.Dropdown(
                choices=['resnet50', 'efficientnet_b0'],
                value='resnet50',
                label="Modelo para Predição"
            )
            
            gr.Markdown("#### Carregar Modelo")
            checkpoint_path = gr.Textbox(
                value="./checkpoints/resnet50/best_model.pth",
                label="Checkpoint"
            )
            load_btn = gr.Button("Carregar Modelo")
            model_status = gr.Textbox(label="Status", interactive=False)
            
            show_gradcam = gr.Checkbox(
                value=True,
                label="Mostrar Grad-CAM",
                info="Visualiza regiões importantes"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### Imagem e Resultados")
            input_image = gr.Image(type="pil", label="Carregar Imagem")
            predict_btn = gr.Button("🔮 Fazer Predição", variant="primary")
            prediction_output = gr.Markdown()
            gradcam_output = gr.Image(label="Grad-CAM")
    
    # Eventos
    def load_wrapper(path, model):
        return prediction_manager.load_model(model, path)
    
    def predict_wrapper(image, model, show_cam):
        if image is None:
            return "Carregue uma imagem primeiro", None
        pred_text, probs, gradcam_img = prediction_manager.predict_single(image, model, show_cam)
        return pred_text, gradcam_img
    
    load_btn.click(fn=load_wrapper, inputs=[checkpoint_path, model_select], outputs=model_status)
    predict_btn.click(fn=predict_wrapper, inputs=[input_image, model_select, show_gradcam], 
                     outputs=[prediction_output, gradcam_output])


def create_batch_prediction_interface(prediction_manager):
    """Cria interface para predição em lote."""
    gr.Markdown("# 📁 Predição em Lote")
    gr.Markdown("Faça predições em múltiplas imagens simultaneamente.")
    
    with gr.Row():
        model_select = gr.Dropdown(
            choices=['resnet50', 'efficientnet_b0'],
            value='resnet50',
            label="Modelo"
        )
        images_input = gr.File(file_count="multiple", file_types=["image"], label="Imagens")
        process_btn = gr.Button("🚀 Processar")
    
    results_output = gr.Dataframe(headers=["Imagem", "Predição", "Prob. Benigno", "Prob. Maligno"])
    
    def process_batch(files, model_name):
        if not files:
            return []
        results = []
        from PIL import Image
        for file in files:
            img = Image.open(file.name)
            _, probs, _ = prediction_manager.predict_single(img, model_name, False)
            if probs:
                pred = 'Maligno' if probs['Maligno'] > probs['Benigno'] else 'Benigno'
                results.append([file.name, pred, f"{probs['Benigno']:.2%}", f"{probs['Maligno']:.2%}"])
        return results
    
    process_btn.click(fn=process_batch, inputs=[images_input, model_select], outputs=results_output)
