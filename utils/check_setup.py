"""
Script para verificar se o ambiente está configurado corretamente.
"""

import sys
from pathlib import Path


def check_python_version():
    """Verifica versão do Python."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requer Python 3.8+)")
        return False


def check_dependencies():
    """Verifica dependências principais."""
    required = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'sklearn',
        'PIL',
        'cv2',
        'albumentations',
        'tensorboard',
        'yaml'
    ]
    
    missing = []
    for dep in required:
        try:
            if dep == 'PIL':
                __import__('PIL')
            elif dep == 'cv2':
                __import__('cv2')
            elif dep == 'sklearn':
                __import__('sklearn')
            elif dep == 'yaml':
                __import__('yaml')
            else:
                __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"❌ {dep} não instalado")
            missing.append(dep)
    
    return len(missing) == 0, missing


def check_dataset_structure(data_dir='./data/isic2020'):
    """Verifica estrutura do dataset."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Dataset não encontrado: {data_dir}")
        print("   Execute: python data/organize_isic.py --help")
        return False
    
    benign_dir = data_path / 'benign'
    malignant_dir = data_path / 'malignant'
    
    if not benign_dir.exists():
        print(f"❌ Pasta 'benign' não encontrada em: {data_dir}")
        return False
    
    if not malignant_dir.exists():
        print(f"❌ Pasta 'malignant' não encontrada em: {data_dir}")
        return False
    
    # Contar imagens
    benign_count = len(list(benign_dir.glob('*.jpg'))) + len(list(benign_dir.glob('*.png')))
    malignant_count = len(list(malignant_dir.glob('*.jpg'))) + len(list(malignant_dir.glob('*.png')))
    
    print(f"✓ Dataset encontrado: {data_dir}")
    print(f"  Benign: {benign_count} imagens")
    print(f"  Malignant: {malignant_count} imagens")
    print(f"  Total: {benign_count + malignant_count} imagens")
    
    if benign_count == 0 or malignant_count == 0:
        print("⚠️  Aviso: Dataset vazio ou desbalanceado")
    
    return True


def check_directories():
    """Verifica se diretórios necessários existem."""
    dirs = [
        './data',
        './checkpoints',
        './results',
        './runs'
    ]
    
    all_exist = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"⚠️  {dir_path} não existe (será criado automaticamente)")
            all_exist = False
    
    return True  # Não é crítico, serão criados automaticamente


def main():
    """Função principal."""
    print("="*60)
    print("VERIFICAÇÃO DE AMBIENTE")
    print("="*60 + "\n")
    
    all_ok = True
    
    print("1. Versão do Python:")
    if not check_python_version():
        all_ok = False
    
    print("\n2. Dependências:")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_ok = False
        print(f"\n   Instale as dependências faltantes:")
        print(f"   pip install {' '.join(missing)}")
    
    print("\n3. Estrutura de diretórios:")
    check_directories()
    
    print("\n4. Dataset:")
    dataset_ok = check_dataset_structure()
    if not dataset_ok:
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ AMBIENTE CONFIGURADO CORRETAMENTE")
        print("="*60)
        print("\nVocê pode executar:")
        print("  python experiments/train_resnet.py")
        print("  python experiments/train_efficientnet.py")
    else:
        print("❌ PROBLEMAS ENCONTRADOS")
        print("="*60)
        print("\nCorrija os problemas acima antes de continuar.")
        sys.exit(1)


if __name__ == '__main__':
    main()
