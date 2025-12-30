"""
Módulo para garantir reprodutibilidade e gerenciar dispositivos.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Define seed para garantir reprodutibilidade em todos os componentes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Retorna dispositivo disponível (CUDA se disponível, senão CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU disponível: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU não disponível, usando CPU")
    
    return device
