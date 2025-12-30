"""
Módulo para criação de modelos EfficientNet-B0.
"""

import torch.nn as nn
from torchvision import models


class EfficientNetB0(nn.Module):
    """
    Wrapper para EfficientNet-B0 adaptado para classificação binária.
    
    Substitui a camada final para o número de classes desejado e expõe
    a camada de features para uso com Grad-CAM.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetB0, self).__init__()
        
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_extractor_layer(self):
        """Retorna última camada do bloco features para uso com Grad-CAM."""
        return self.backbone.features[-1]


def get_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> EfficientNetB0:
    """Factory function para criar instância de EfficientNet-B0."""
    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)
