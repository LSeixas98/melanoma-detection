"""
Script para organizar imagens do dataset ISIC em pastas benign/ e malignant/
baseado em arquivos CSV com metadados.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional


def organize_isic_dataset(
    images_dir: str,
    csv_path: str,
    target_dir: str,
    image_col: str = 'image_name',
    label_col: str = 'benign_malignant',
    ground_truth_csv: Optional[str] = None,
    gt_image_col: Optional[str] = None,
    gt_label_col: Optional[str] = None,
    copy_files: bool = True
):
    """
    Organiza imagens do ISIC em pastas benign/ e malignant/ baseado em CSV.
    
    Args:
        images_dir: Diretório onde estão todas as imagens do ISIC
        csv_path: Caminho para o arquivo CSV com metadados
        target_dir: Diretório onde serão organizadas as imagens (data/isic2020/)
        image_col: Nome da coluna no CSV com o nome da imagem
        label_col: Nome da coluna no CSV com a label (0=benign, 1=malignant)
        ground_truth_csv: Caminho opcional para arquivo CSV separado com labels (ground truth)
        gt_image_col: Nome da coluna de imagem no ground truth CSV (se diferente)
        gt_label_col: Nome da coluna de label no ground truth CSV (se diferente)
        copy_files: Se True, copia os arquivos. Se False, move os arquivos
    """
    images_path = Path(images_dir)
    target_path = Path(target_dir)
    csv_file = Path(csv_path)
    
    # Criar estrutura de diretórios
    benign_dir = target_path / 'benign'
    malignant_dir = target_path / 'malignant'
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    
    # Ler CSV de metadados
    print(f"📄 Lendo CSV de metadados: {csv_path}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        return
    
    print(f"✓ CSV carregado: {len(df)} entradas")
    print(f"  Colunas disponíveis: {list(df.columns)}")
    
    # Verificar se a coluna de imagem existe
    if image_col not in df.columns:
        print(f"❌ Coluna '{image_col}' não encontrada no CSV")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        print(f"   💡 Dica: Use --image-col para especificar o nome correto da coluna")
        return
    
    # Se não tem label no CSV de metadados, tentar usar ground truth
    if label_col not in df.columns:
        if ground_truth_csv is None:
            print(f"⚠️  Coluna '{label_col}' não encontrada no CSV de metadados")
            print(f"   Colunas disponíveis: {list(df.columns)}")
            print(f"   💡 Você precisa fornecer um arquivo de ground truth com --ground-truth")
            return
        else:
            # Ler arquivo de ground truth
            print(f"\n📄 Lendo arquivo de ground truth: {ground_truth_csv}")
            try:
                gt_df = pd.read_csv(ground_truth_csv)
                print(f"✓ Ground truth carregado: {len(gt_df)} entradas")
                print(f"  Colunas disponíveis: {list(gt_df.columns)}")
                
                # Determinar nomes das colunas no ground truth
                gt_img_col = gt_image_col if gt_image_col else image_col
                gt_lbl_col = gt_label_col if gt_label_col else label_col
                
                # Verificar se as colunas existem no ground truth
                if gt_img_col not in gt_df.columns:
                    print(f"❌ Coluna '{gt_img_col}' não encontrada no ground truth")
                    print(f"   Colunas disponíveis: {list(gt_df.columns)}")
                    return
                
                if gt_lbl_col not in gt_df.columns:
                    print(f"❌ Coluna '{gt_lbl_col}' não encontrada no ground truth")
                    print(f"   Colunas disponíveis: {list(gt_df.columns)}")
                    return
                
                # Fazer merge dos dataframes
                print(f"\n🔗 Fazendo merge dos CSVs...")
                df = df.merge(
                    gt_df[[gt_img_col, gt_lbl_col]],
                    left_on=image_col,
                    right_on=gt_img_col,
                    how='inner'
                )
                # Renomear coluna de label para padronizar
                df = df.rename(columns={gt_lbl_col: label_col})
                print(f"✓ Merge concluído: {len(df)} imagens com labels")
                
            except Exception as e:
                print(f"❌ Erro ao ler ground truth CSV: {e}")
                return
    
    # Verificar se agora temos a coluna de label
    if label_col not in df.columns:
        print(f"❌ Coluna '{label_col}' não encontrada após merge")
        return
    
    # Contadores
    benign_count = 0
    malignant_count = 0
    not_found = 0
    already_exists = 0
    
    print(f"\n📋 Organizando imagens...")
    print(f"  Origem: {images_dir}")
    print(f"  Destino: {target_dir}")
    print(f"  Modo: {'Cópia' if copy_files else 'Mover'}")
    
    # Processar cada linha do CSV
    for idx, row in df.iterrows():
        image_name = str(row[image_col]).strip()
        label = row[label_col]
        
        # Determinar se é benign ou malignant
        # Pode ser 0/1, 'benign'/'malignant', ou outros formatos
        if pd.isna(label):
            continue
            
        # Normalizar label
        if isinstance(label, (int, float)):
            is_malignant = int(label) == 1
        elif isinstance(label, str):
            label_lower = label.lower().strip()
            is_malignant = label_lower in ['1', 'malignant', 'maligno', 'yes', 'true']
        else:
            continue
        
        # Procurar imagem (pode ter diferentes extensões)
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_file = images_path / f"{image_name}{ext}"
            if potential_file.exists():
                image_file = potential_file
                break
        
        if image_file is None:
            # Tentar sem extensão no nome (o nome já pode incluir extensão)
            image_name_no_ext = image_name.rsplit('.', 1)[0] if '.' in image_name else image_name
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_file = images_path / f"{image_name_no_ext}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
        
        if image_file is None:
            # Tentar buscar por padrão (caso o nome no CSV seja diferente do arquivo)
            image_name_clean = image_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            for ext_file in images_path.glob(f"{image_name_clean}.*"):
                if ext_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_file = ext_file
                    break
        
        if image_file is None:
            not_found += 1
            if not_found <= 5:  # Mostrar apenas os primeiros 5
                print(f"  ⚠️  Imagem não encontrada: {image_name}")
            continue
        
        # Determinar pasta destino
        dest_dir = malignant_dir if is_malignant else benign_dir
        dest_file = dest_dir / image_file.name
        
        # Verificar se já existe
        if dest_file.exists():
            already_exists += 1
            continue
        
        # Copiar ou mover
        try:
            if copy_files:
                shutil.copy2(image_file, dest_file)
            else:
                shutil.move(str(image_file), str(dest_file))
            
            if is_malignant:
                malignant_count += 1
            else:
                benign_count += 1
            
            # Progresso a cada 1000 imagens
            total = benign_count + malignant_count
            if total % 1000 == 0:
                print(f"  ✓ Processadas: {total} imagens...")
                
        except Exception as e:
            print(f"  ❌ Erro ao processar {image_name}: {e}")
    
    # Resumo
    print(f"\n✅ Organização concluída!")
    print(f"  📊 Estatísticas:")
    print(f"     Benign: {benign_count} imagens")
    print(f"     Malignant: {malignant_count} imagens")
    print(f"     Total: {benign_count + malignant_count} imagens")
    if not_found > 0:
        print(f"     ⚠️  Não encontradas: {not_found} imagens")
    if already_exists > 0:
        print(f"     ℹ️  Já existiam: {already_exists} imagens")
    
    print(f"\n  📁 Estrutura criada:")
    print(f"     {target_dir}/")
    print(f"       ├── benign/ ({benign_count} imagens)")
    print(f"       └── malignant/ ({malignant_count} imagens)")
    
    return benign_count, malignant_count


def organize_multiple_csvs(
    images_dir: str,
    csv_files: list,
    target_dir: str,
    image_col: str = 'image_name',
    label_col: str = 'benign_malignant',
    ground_truth_csv: Optional[str] = None,
    gt_image_col: Optional[str] = None,
    gt_label_col: Optional[str] = None,
    copy_files: bool = True
):
    """
    Organiza imagens usando múltiplos arquivos CSV (ex: Train e Test).
    
    Args:
        images_dir: Diretório onde estão todas as imagens do ISIC
        csv_files: Lista de caminhos para arquivos CSV
        target_dir: Diretório onde serão organizadas as imagens
        image_col: Nome da coluna no CSV com o nome da imagem
        label_col: Nome da coluna no CSV com a label
        copy_files: Se True, copia os arquivos. Se False, move os arquivos
    """
    total_benign = 0
    total_malignant = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*60}")
        print(f"Processando CSV {i}/{len(csv_files)}: {Path(csv_file).name}")
        print(f"{'='*60}")
        
        benign, malignant = organize_isic_dataset(
            images_dir=images_dir,
            csv_path=csv_file,
            target_dir=target_dir,
            image_col=image_col,
            label_col=label_col,
            ground_truth_csv=ground_truth_csv,
            gt_image_col=gt_image_col,
            gt_label_col=gt_label_col,
            copy_files=copy_files
        )
        
        total_benign += benign
        total_malignant += malignant
    
    print(f"\n{'='*60}")
    print(f"✅ Processamento completo de todos os CSVs!")
    print(f"  📊 Total geral:")
    print(f"     Benign: {total_benign} imagens")
    print(f"     Malignant: {total_malignant} imagens")
    print(f"     Total: {total_benign + total_malignant} imagens")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Organizar imagens do dataset ISIC em pastas benign/ e malignant/'
    )
    parser.add_argument('--images', type=str, required=True,
                        help='Diretório onde estão todas as imagens do ISIC')
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                        help='Caminho(s) para arquivo(s) CSV com metadados (ex: ISIC_2020_Train_Metadata.csv ISIC_2020_Test_Metadata.csv)')
    parser.add_argument('--target', type=str, default='./data/isic2020',
                        help='Diretório onde serão organizadas as imagens (padrão: ./data/isic2020)')
    parser.add_argument('--image-col', type=str, default='image_name',
                        help='Nome da coluna no CSV com o nome da imagem (padrão: image_name)')
    parser.add_argument('--label-col', type=str, default='benign_malignant',
                        help='Nome da coluna no CSV com a label (padrão: benign_malignant)')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Caminho para arquivo CSV separado com labels (ground truth) - necessário se o CSV de metadados não tiver labels')
    parser.add_argument('--gt-image-col', type=str, default=None,
                        help='Nome da coluna de imagem no ground truth CSV (padrão: usa --image-col)')
    parser.add_argument('--gt-label-col', type=str, default=None,
                        help='Nome da coluna de label no ground truth CSV (padrão: usa --label-col)')
    parser.add_argument('--move', action='store_true',
                        help='Mover arquivos ao invés de copiar (padrão: copiar)')
    
    args = parser.parse_args()
    
    if len(args.csv) == 1:
        # Processar apenas um CSV
        organize_isic_dataset(
            images_dir=args.images,
            csv_path=args.csv[0],
            target_dir=args.target,
            image_col=args.image_col,
            label_col=args.label_col,
            ground_truth_csv=args.ground_truth,
            gt_image_col=args.gt_image_col,
            gt_label_col=args.gt_label_col,
            copy_files=not args.move
        )
    else:
        # Processar múltiplos CSVs
        organize_multiple_csvs(
            images_dir=args.images,
            csv_files=args.csv,
            target_dir=args.target,
            image_col=args.image_col,
            label_col=args.label_col,
            ground_truth_csv=args.ground_truth,
            gt_image_col=args.gt_image_col,
            gt_label_col=args.gt_label_col,
            copy_files=not args.move
        )
