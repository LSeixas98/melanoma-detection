"""
Módulo para criação de modelos ResNet-50.
"""

import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    """
    Wrapper para ResNet-50 adaptado para classificação binária.
    
    Substitui a camada final para o número de classes desejado e expõe
    a camada de features para uso com Grad-CAM.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(ResNet50, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_extractor_layer(self):
        """Retorna layer4 para uso com Grad-CAM (última camada convolucional)."""
        return self.backbone.layer4


def get_resnet50(num_classes: int = 2, pretrained: bool = True) -> ResNet50:
    """Factory function para criar instância de ResNet-50."""
    return ResNet50(num_classes=num_classes, pretrained=pretrained)
