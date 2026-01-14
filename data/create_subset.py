"""
Script para criar um subconjunto menor do dataset para testes rápidos.
Mantém a proporção original de classes (benign/malignant).
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple


def create_subset_dataset(
    source_dir: str,
    target_dir: str,
    subset_size: float = 0.1,  # 10% do dataset original
    seed: int = 42
):
    """
    Cria um subconjunto menor do dataset mantendo a proporção de classes.
    
    Args:
        source_dir: Diretório do dataset completo (deve ter pastas benign/ e malignant/)
        target_dir: Diretório onde será criado o subconjunto
        subset_size: Proporção do dataset a ser copiada (0.1 = 10%, 0.05 = 5%)
        seed: Seed para reprodutibilidade
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Criar estrutura de diretórios
    benign_source = source_path / 'benign'
    malignant_source = source_path / 'malignant'
    
    benign_target = target_path / 'benign'
    malignant_target = target_path / 'malignant'
    
    benign_target.mkdir(parents=True, exist_ok=True)
    malignant_target.mkdir(parents=True, exist_ok=True)
    
    # Listar todas as imagens
    benign_images = [f for f in benign_source.iterdir() 
                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    malignant_images = [f for f in malignant_source.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"📊 Dataset original:")
    print(f"  Benign: {len(benign_images)} imagens")
    print(f"  Malignant: {len(malignant_images)} imagens")
    print(f"  Total: {len(benign_images) + len(malignant_images)} imagens")
    
    # Calcular quantas imagens copiar
    num_benign = max(1, int(len(benign_images) * subset_size))
    num_malignant = max(1, int(len(malignant_images) * subset_size))
    
    # Selecionar amostra aleatória
    selected_benign = random.sample(benign_images, num_benign)
    selected_malignant = random.sample(malignant_images, num_malignant)
    
    print(f"\n📦 Criando subconjunto ({subset_size*100:.1f}%):")
    print(f"  Benign: {num_benign} imagens")
    print(f"  Malignant: {num_malignant} imagens")
    print(f"  Total: {num_benign + num_malignant} imagens")
    
    # Copiar imagens
    print(f"\n📋 Copiando imagens...")
    for img in selected_benign:
        shutil.copy2(img, benign_target / img.name)
    
    for img in selected_malignant:
        shutil.copy2(img, malignant_target / img.name)
    
    print(f"\n✅ Subconjunto criado em: {target_dir}")
    print(f"   Estrutura:")
    print(f"   {target_dir}/")
    print(f"     ├── benign/ ({num_benign} imagens)")
    print(f"     └── malignant/ ({num_malignant} imagens)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Criar subconjunto do dataset para testes')
    parser.add_argument('--source', type=str, required=True,
                        help='Diretório do dataset completo (com pastas benign/ e malignant/)')
    parser.add_argument('--target', type=str, default='./data/isic2020_test',
                        help='Diretório onde será criado o subconjunto (padrão: ./data/isic2020_test)')
    parser.add_argument('--size', type=float, default=0.1,
                        help='Proporção do dataset a copiar (0.1 = 10%%, 0.05 = 5%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para reprodutibilidade')
    
    args = parser.parse_args()
    
    create_subset_dataset(
        source_dir=args.source,
        target_dir=args.target,
        subset_size=args.size,
        seed=args.seed
    )
