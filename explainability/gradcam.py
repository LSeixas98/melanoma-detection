"""
Implementação de Grad-CAM e Grad-CAM++ para explicabilidade.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Gera mapas de atenção visual destacando regiões importantes para decisão.
    
    Referência:
        Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from 
        Deep Networks via Gradient-based Localization
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Modelo PyTorch
            target_layer: Camada convolucional alvo (última layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook para salvar ativações do forward pass."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook para salvar gradientes do backward pass."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                    target_class: Optional[int] = None) -> np.ndarray:
        """
        Gera mapa de atenção Grad-CAM.
        
        Args:
            input_tensor: Imagem de entrada (1, C, H, W)
            target_class: Classe alvo (None = classe predita)
            
        Returns:
            Mapa de calor normalizado [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Se target_class não especificada, usar classe predita
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zerar gradientes
        self.model.zero_grad()
        
        # Backward pass para classe alvo
        class_score = output[0, target_class]
        class_score.backward()
        
        # Pesos: global average pooling dos gradientes
        # Shape: (batch, channels, H, W) -> (channels,)
        weights = self.gradients.mean(dim=(2, 3))[0]
        
        # Combinação linear ponderada das ativações
        # Shape: (channels, H, W)
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]
        
        # Aplicar ReLU (considerar apenas influências positivas)
        cam = F.relu(cam)
        
        # Normalizar para [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self, input_image: np.ndarray, cam: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
        """
        Sobrepõe mapa de calor na imagem original.
        
        Args:
            input_image: Imagem original (H, W, 3) em [0, 255] RGB
            cam: Mapa Grad-CAM (H_cam, W_cam) em [0, 1]
            alpha: Transparência do heatmap (0=invisível, 1=opaco)
            
        Returns:
            Imagem com heatmap sobreposto (H, W, 3)
        """
        # Redimensionar CAM para tamanho da imagem original
        h, w = input_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Converter CAM para heatmap colorido
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Sobrepor heatmap na imagem
        superimposed = heatmap * alpha + input_image * (1 - alpha)
        superimposed = np.uint8(superimposed)
        
        return superimposed


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ - versão aprimorada com ponderação pixel-wise.
    
    Referência:
        Chattopadhay et al. (2018) - Grad-CAM++: Generalized Gradient-Based 
        Visual Explanations for Deep Convolutional Networks
    """
    
    def generate_cam(self, input_tensor: torch.Tensor,
                    target_class: Optional[int] = None) -> np.ndarray:
        """
        Gera mapa Grad-CAM++ com ponderação aprimorada.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()
        
        # Gradientes e ativações
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Ponderação alpha (Grad-CAM++)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # Evitar divisão por zero
        alpha = grad_2 / (2 * grad_2 + (activations * grad_3).sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        # Pesos com ponderação positiva
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Combinação linear
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU e normalização
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def compare_gradcam_methods(model, input_tensor, input_image, target_layer,
                           target_class=None, save_path=None):
    """
    Compara Grad-CAM e Grad-CAM++ lado a lado.
    
    Args:
        model: Modelo PyTorch
        input_tensor: Tensor de entrada (1, 3, H, W)
        input_image: Imagem original numpy (H, W, 3) RGB
        target_layer: Camada convolucional alvo
        target_class: Classe alvo (None = predita)
        save_path: Caminho para salvar figura (None = apenas exibir)
    """
    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, target_class)
    viz_gradcam = gradcam.visualize(input_image, cam)
    
    # Grad-CAM++
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    cam_pp = gradcam_pp.generate_cam(input_tensor, target_class)
    viz_gradcam_pp = gradcam_pp.visualize(input_image, cam_pp)
    
    # Plotar comparação
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_image)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(viz_gradcam)
    axes[1].set_title('Grad-CAM', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(viz_gradcam_pp)
    axes[2].set_title('Grad-CAM++', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparação salva em: {save_path}")
    else:
        plt.show()
    
    return viz_gradcam, viz_gradcam_pp