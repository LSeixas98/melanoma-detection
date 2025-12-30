"""
Módulo para carregamento e divisão do dataset ISIC 2020.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


class ISIC2020Dataset(Dataset):
    """
    Dataset para ISIC 2020 Challenge com suporte a Albumentations.
    
    Estrutura esperada:
        data_dir/
            benign/
            malignant/
    """
    
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        self.samples = []
        self.targets = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
                        self.targets.append(class_idx)
        
        print(f"✓ Dataset carregado de: {root}")
        print(f"  Total de imagens: {len(self)}")
        print(f"  Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.targets[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            try:
                import albumentations as A
                if isinstance(self.transform, A.core.composition.Compose):
                    image_np = np.array(image)
                    transformed = self.transform(image=image_np)
                    image = transformed['image']
                else:
                    image = self.transform(image)
            except (ImportError, AttributeError):
                image = self.transform(image)
        
        return image, label


def get_dataloaders(config: dict, train_transform, val_transform) -> Tuple:
    """
    Cria DataLoaders para treino, validação e teste com divisão estratificada.
    
    Divisão padrão: 70% treino, 15% validação, 15% teste.
    """
    from data.preprocessing import get_weighted_sampler
    
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    seed = config['random_seed']
    
    full_dataset = ISIC2020Dataset(data_dir, transform=None)
    
    indices = list(range(len(full_dataset)))
    labels = full_dataset.targets
    
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config['data']['test_split'],
        stratify=labels,
        random_state=seed
    )
    
    train_val_labels = [labels[i] for i in train_val_idx]
    val_size_adjusted = config['data']['val_split'] / (1 - config['data']['test_split'])
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=seed
    )
    
    print(f"\n✓ Divisão do dataset:")
    print(f"  Treino: {len(train_idx)} imagens ({len(train_idx)/len(full_dataset)*100:.1f}%)")
    print(f"  Validação: {len(val_idx)} imagens ({len(val_idx)/len(full_dataset)*100:.1f}%)")
    print(f"  Teste: {len(test_idx)} imagens ({len(test_idx)/len(full_dataset)*100:.1f}%)")
    
    train_dataset = ISIC2020Dataset(data_dir, transform=train_transform)
    train_subset = Subset(train_dataset, train_idx)
    
    val_dataset = ISIC2020Dataset(data_dir, transform=val_transform)
    val_subset = Subset(val_dataset, val_idx)
    
    test_dataset = ISIC2020Dataset(data_dir, transform=val_transform)
    test_subset = Subset(test_dataset, test_idx)
    
    train_labels = [labels[i] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor([1.0, class_counts[0] / class_counts[1]], dtype=torch.float32)
    
    temp_train_dataset = type('obj', (object,), {'targets': train_labels})()
    train_sampler = get_weighted_sampler(temp_train_dataset)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders criados (batch_size={batch_size})")
    return train_loader, val_loader, test_loader, class_weights
