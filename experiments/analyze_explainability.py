"""
Script para análise de explicabilidade usando Grad-CAM em múltiplas imagens.

Gera mapas de atenção para comparação visual entre ResNet-50 e EfficientNet-B0.

Uso:
    python experiments/analyze_explainability.py --num_samples 100
"""

import sys
sys.path.append('.')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils.reproducibility import set_seed, get_device
from utils.config import DEFAULT_CONFIG
from data.preprocessing import get_transforms
from data.dataset import get_dataloaders
from models.resnet import get_resnet50
from models.efficientnet import get_efficientnet_b0
from explainability.gradcam import GradCAM, GradCAMPlusPlus, compare_gradcam_methods


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Desnormaliza tensor para visualização.
    
    Args:
        tensor: Tensor normalizado (C, H, W)
        mean: Média usada na normalização
        std: Desvio padrão usado na normalização
        
    Returns:
        Array numpy (H, W, C) em [0, 255]
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clampar para [0, 1] e converter para numpy
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def analyze_sample(image_tensor, label, resnet_model, effnet_model,
                   resnet_layer, effnet_layer, device, save_dir):
    """
    Analisa uma amostra individual com ambos os modelos.
    
    Args:
        image_tensor: Tensor da imagem (C, H, W)
        label: Rótulo verdadeiro
        resnet_model: Modelo ResNet-50
        effnet_model: Modelo EfficientNet-B0
        resnet_layer: Camada alvo ResNet
        effnet_layer: Camada alvo EfficientNet
        device: Dispositivo
        save_dir: Diretório para salvar
        
    Returns:
        Dicionário com informações da análise
    """
    # Preparar input
    input_tensor = image_tensor.unsqueeze(0).to(device)
    original_image = denormalize_image(image_tensor)
    
    # Predições
    resnet_model.eval()
    effnet_model.eval()
    
    with torch.no_grad():
        resnet_output = resnet_model(input_tensor)
        effnet_output = effnet_model(input_tensor)
        
        resnet_probs = torch.softmax(resnet_output, dim=1)
        effnet_probs = torch.softmax(effnet_output, dim=1)
        
        resnet_pred = resnet_output.argmax(dim=1).item()
        effnet_pred = effnet_output.argmax(dim=1).item()
    
    # Grad-CAM para ambos modelos
    gradcam_resnet = GradCAM(resnet_model, resnet_layer)
    cam_resnet = gradcam_resnet.generate_cam(input_tensor)
    viz_resnet = gradcam_resnet.visualize(original_image, cam_resnet)
    
    gradcam_effnet = GradCAM(effnet_model, effnet_layer)
    cam_effnet = gradcam_effnet.generate_cam(input_tensor)
    viz_effnet = gradcam_effnet.visualize(original_image, cam_effnet)
    
    # Criar visualização comparativa
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original\nLabel: {"Maligno" if label == 1 else "Benigno"}',
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # ResNet-50
    axes[1].imshow(viz_resnet)
    pred_text = "Maligno" if resnet_pred == 1 else "Benigno"
    conf = resnet_probs[0, resnet_pred].item() * 100
    color = 'green' if resnet_pred == label else 'red'
    axes[1].set_title(f'ResNet-50\nPred: {pred_text} ({conf:.1f}%)',
                     fontsize=12, fontweight='bold', color=color)
    axes[1].axis('off')
    
    # EfficientNet-B0
    axes[2].imshow(viz_effnet)
    pred_text = "Maligno" if effnet_pred == 1 else "Benigno"
    conf = effnet_probs[0, effnet_pred].item() * 100
    color = 'green' if effnet_pred == label else 'red'
    axes[2].set_title(f'EfficientNet-B0\nPred: {pred_text} ({conf:.1f}%)',
                     fontsize=12, fontweight='bold', color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return {
        'label': label,
        'resnet_pred': resnet_pred,
        'effnet_pred': effnet_pred,
        'resnet_conf': resnet_probs[0, resnet_pred].item(),
        'effnet_conf': effnet_probs[0, effnet_pred].item(),
        'fig': fig,
        'original': original_image,
        'viz_resnet': viz_resnet,
        'viz_effnet': viz_effnet
    }


def main(args):
    """
    Função principal de análise.
    """
    print("\n" + "="*60)
    print("ANÁLISE DE EXPLICABILIDADE - GRAD-CAM")
    print("="*60 + "\n")
    
    # Setup
    set_seed(42)
    device = get_device()
    config = DEFAULT_CONFIG.copy()
    
    # Criar diretório de saída
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    print("[1/5] Carregando dados...")
    val_transform = get_transforms(config, train=False)
    _, _, test_loader, _ = get_dataloaders(
        config, val_transform, val_transform
    )
    
    # Carregar modelos
    print("\n[2/5] Carregando modelos...")
    
    resnet_model = get_resnet50(num_classes=2, pretrained=True).to(device)
    resnet_checkpoint = Path(args.resnet_checkpoint)
    if resnet_checkpoint.exists():
        checkpoint = torch.load(resnet_checkpoint, map_location=device)
        resnet_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ ResNet-50 carregado: {resnet_checkpoint}")
    else:
        print(f"⚠ Checkpoint ResNet não encontrado, usando pré-treinado ImageNet")
    
    effnet_model = get_efficientnet_b0(num_classes=2, pretrained=True).to(device)
    effnet_checkpoint = Path(args.effnet_checkpoint)
    if effnet_checkpoint.exists():
        checkpoint = torch.load(effnet_checkpoint, map_location=device)
        effnet_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ EfficientNet-B0 carregado: {effnet_checkpoint}")
    else:
        print(f"⚠ Checkpoint EfficientNet não encontrado, usando pré-treinado ImageNet")
    
    # Obter camadas alvo
    resnet_layer = resnet_model.get_feature_extractor_layer()
    effnet_layer = effnet_model.get_feature_extractor_layer()
    
    # Selecionar amostras
    print(f"\n[3/5] Selecionando {args.num_samples} amostras...")
    
    all_images = []
    all_labels = []
    
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if len(all_images) * images.size(0) >= args.num_samples:
            break
    
    all_images = torch.cat(all_images, dim=0)[:args.num_samples]
    all_labels = torch.cat(all_labels, dim=0)[:args.num_samples]
    
    # Selecionar balanceado (metade benigno, metade maligno se possível)
    benign_indices = (all_labels == 0).nonzero(as_tuple=True)[0]
    malignant_indices = (all_labels == 1).nonzero(as_tuple=True)[0]
    
    num_per_class = args.num_samples // 2
    selected_benign = benign_indices[:min(num_per_class, len(benign_indices))]
    selected_malignant = malignant_indices[:min(num_per_class, len(malignant_indices))]
    
    selected_indices = torch.cat([selected_benign, selected_malignant])
    
    print(f"  Selecionado: {len(selected_benign)} benignas, {len(selected_malignant)} malignas")
    
    # Analisar amostras
    print(f"\n[4/5] Gerando mapas Grad-CAM para {len(selected_indices)} amostras...")
    
    results = []
    agreements = []
    
    for idx in tqdm(selected_indices):
        image = all_images[idx]
        label = all_labels[idx].item()
        
        result = analyze_sample(
            image, label,
            resnet_model, effnet_model,
            resnet_layer, effnet_layer,
            device, save_dir
        )
        
        results.append(result)
        
        # Concordância entre modelos
        if result['resnet_pred'] == result['effnet_pred']:
            agreements.append(1)
        else:
            agreements.append(0)
        
        # Salvar figura
        sample_id = len(results)
        filename = f"sample_{sample_id:03d}_label{label}_resnet{result['resnet_pred']}_effnet{result['effnet_pred']}.png"
        result['fig'].savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(result['fig'])
    
    # Análise agregada
    print(f"\n[5/5] Análise agregada...")
    
    total = len(results)
    agreement_rate = sum(agreements) / total * 100
    
    resnet_correct = sum(1 for r in results if r['resnet_pred'] == r['label'])
    effnet_correct = sum(1 for r in results if r['effnet_pred'] == r['label'])
    
    resnet_acc = resnet_correct / total * 100
    effnet_acc = effnet_correct / total * 100
    
    print(f"\n{'='*60}")
    print("RESUMO DA ANÁLISE")
    print(f"{'='*60}")
    print(f"Total de amostras analisadas: {total}")
    print(f"\nAcurácia nas amostras:")
    print(f"  ResNet-50:       {resnet_correct}/{total} ({resnet_acc:.1f}%)")
    print(f"  EfficientNet-B0: {effnet_correct}/{total} ({effnet_acc:.1f}%)")
    print(f"\nConcordância entre modelos: {sum(agreements)}/{total} ({agreement_rate:.1f}%)")
    print(f"\nVisualiz ações salvas em: {save_dir}")
    print(f"{'='*60}\n")
    
    # Criar índice HTML para visualização fácil
    create_html_index(results, save_dir, agreement_rate, resnet_acc, effnet_acc)
    
    print(f"✓ Análise completa! Abra {save_dir}/index.html para visualizar")


def create_html_index(results, save_dir, agreement_rate, resnet_acc, effnet_acc):
    """
    Cria índice HTML com todas as visualizações.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Análise de Explicabilidade - Grad-CAM</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }}
            .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}
            .sample {{ background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .sample img {{ width: 100%; border-radius: 5px; }}
            .correct {{ border-left: 5px solid green; }}
            .incorrect {{ border-left: 5px solid red; }}
            .disagreement {{ border-left: 5px solid orange; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Análise de Explicabilidade - Grad-CAM</h1>
            <p>Comparação de mapas de atenção entre ResNet-50 e EfficientNet-B0</p>
            <div class="stats">
                <div class="stat-box">
                    <h3>ResNet-50</h3>
                    <p>Acurácia: {resnet_acc:.1f}%</p>
                </div>
                <div class="stat-box">
                    <h3>EfficientNet-B0</h3>
                    <p>Acurácia: {effnet_acc:.1f}%</p>
                </div>
                <div class="stat-box">
                    <h3>Concordância</h3>
                    <p>{agreement_rate:.1f}%</p>
                </div>
            </div>
        </div>
        
        <div class="gallery">
    """
    
    for i, result in enumerate(results, 1):
        label_text = "Maligno" if result['label'] == 1 else "Benigno"
        resnet_pred_text = "Maligno" if result['resnet_pred'] == 1 else "Benigno"
        effnet_pred_text = "Maligno" if result['effnet_pred'] == 1 else "Benigno"
        
        # Determinar classe CSS
        if result['resnet_pred'] != result['effnet_pred']:
            css_class = "disagreement"
            status = "❌ Discordância"
        elif result['resnet_pred'] == result['label']:
            css_class = "correct"
            status = "✅ Ambos corretos"
        else:
            css_class = "incorrect"
            status = "❌ Ambos incorretos"
        
        filename = f"sample_{i:03d}_label{result['label']}_resnet{result['resnet_pred']}_effnet{result['effnet_pred']}.png"
        
        html_content += f"""
            <div class="sample {css_class}">
                <h3>Amostra {i} - {status}</h3>
                <p><strong>Label Verdadeiro:</strong> {label_text}</p>
                <p><strong>ResNet-50:</strong> {resnet_pred_text} ({result['resnet_conf']*100:.1f}%)</p>
                <p><strong>EfficientNet-B0:</strong> {effnet_pred_text} ({result['effnet_conf']*100:.1f}%)</p>
                <img src="{filename}" alt="Sample {i}">
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(save_dir / 'index.html', 'w') as f:
        f.write(html_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análise de explicabilidade com Grad-CAM')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Número de amostras para analisar')
    parser.add_argument('--resnet_checkpoint', type=str,
                       default='./checkpoints/resnet50/best_model.pth',
                       help='Caminho para checkpoint ResNet-50')
    parser.add_argument('--effnet_checkpoint', type=str,
                       default='./checkpoints/efficientnet_b0/best_model.pth',
                       help='Caminho para checkpoint EfficientNet-B0')
    parser.add_argument('--save_dir', type=str, default='./results/explainability',
                       help='Diretório para salvar visualizações')
    
    args = parser.parse_args()
    main(args)