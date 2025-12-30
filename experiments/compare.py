"""
Script para comparação completa entre ResNet-50 e EfficientNet-B0.

Uso:
    python experiments/compare.py
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc

from utils.reproducibility import set_seed, get_device
from utils.config import DEFAULT_CONFIG
from data.preprocessing import get_transforms
from data.dataset import get_dataloaders
from models.resnet import get_resnet50
from models.efficientnet import get_efficientnet_b0
from evaluation.metrics import evaluate_model, print_metrics, get_predictions
from evaluation.efficiency import benchmark_model, print_efficiency_comparison


def load_trained_model(model, checkpoint_path, device):
    """
    Carrega pesos de modelo treinado.
    
    Args:
        model: Instância do modelo
        checkpoint_path: Caminho para checkpoint
        device: Dispositivo
        
    Returns:
        Modelo com pesos carregados
    """
    if not Path(checkpoint_path).exists():
        print(f"⚠ Checkpoint não encontrado: {checkpoint_path}")
        print("  Usando modelo pré-treinado do ImageNet")
        return model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Modelo carregado: {checkpoint_path}")
    print(f"  Época: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Melhor AUC: {checkpoint.get('best_val_auc', 'N/A'):.4f}")
    
    return model


def plot_roc_curves(y_true_resnet, y_proba_resnet, auc_resnet,
                   y_true_effnet, y_proba_effnet, auc_effnet,
                   save_path='results/roc_comparison.png'):
    """
    Plota curvas ROC comparativas.
    """
    fpr_resnet, tpr_resnet, _ = roc_curve(y_true_resnet, y_proba_resnet)
    fpr_effnet, tpr_effnet, _ = roc_curve(y_true_effnet, y_proba_effnet)
    
    plt.figure(figsize=(10, 8))
    
    # Curva ResNet-50
    plt.plot(fpr_resnet, tpr_resnet, 'b-', linewidth=2,
            label=f'ResNet-50 (AUC = {auc_resnet:.4f})')
    
    # Curva EfficientNet-B0
    plt.plot(fpr_effnet, tpr_effnet, 'r-', linewidth=2,
            label=f'EfficientNet-B0 (AUC = {auc_effnet:.4f})')
    
    # Linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório (AUC = 0.5)')
    
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
    plt.title('Curvas ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Curvas ROC salvas em: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_resnet, metrics_effnet, 
                           save_path='results/metrics_comparison.png'):
    """
    Plota comparação de métricas em gráfico de barras.
    """
    metrics_names = ['Acurácia', 'Sensibilidade', 'Especificidade', 'Precisão', 'F1-Score']
    metrics_keys = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    
    resnet_values = [metrics_resnet[k] for k in metrics_keys]
    effnet_values = [metrics_effnet[k] for k in metrics_keys]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, resnet_values, width, label='ResNet-50', color='#3498db')
    bars2 = ax.bar(x + width/2, effnet_values, width, label='EfficientNet-B0', color='#e74c3c')
    
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparação de Métricas Clínicas', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparação de métricas salva em: {save_path}")
    plt.close()


def statistical_comparison(metrics_resnet, metrics_effnet, y_true_resnet, y_pred_resnet, 
                          y_true_effnet, y_pred_effnet):
    """
    Realiza comparação estatística robusta entre modelos.
    
    Inclui:
    - Teste de McNemar para comparar modelos pareados
    - Intervalos de confiança
    - Análise de significância estatística
    """
    from scipy import stats
    
    print("\n" + "="*60)
    print("ANÁLISE ESTATÍSTICA COMPARATIVA")
    print("="*60)
    
    metrics_keys = ['accuracy', 'sensitivity', 'specificity', 'auc_roc']
    
    print("\nDiferenças entre modelos:")
    print(f"{'Métrica':<20} {'ResNet-50':>12} {'EfficientNet':>12} {'Diferença':>12} {'IC 95%':>20}")
    print("-" * 80)
    
    n = len(y_true_resnet)
    
    for key in metrics_keys:
        resnet_val = metrics_resnet[key]
        effnet_val = metrics_effnet[key]
        diff = effnet_val - resnet_val
        diff_pct = (diff / resnet_val) * 100 if resnet_val > 0 else 0
        
        se_resnet = np.sqrt(resnet_val * (1 - resnet_val) / n)
        se_effnet = np.sqrt(effnet_val * (1 - effnet_val) / n)
        se_diff = np.sqrt(se_resnet**2 + se_effnet**2)
        
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        print(f"{key:<20} {resnet_val:>12.4f} {effnet_val:>12.4f} {diff:>+11.4f} ({diff_pct:+.2f}%) [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("\n" + "-" * 80)
    print("TESTE DE MCNEMAR (Comparação de Modelos Pareados)")
    print("-" * 80)
    
    both_correct = np.sum((y_pred_resnet == y_true_resnet) & (y_pred_effnet == y_true_effnet))
    both_wrong = np.sum((y_pred_resnet != y_true_resnet) & (y_pred_effnet != y_true_effnet))
    resnet_correct_effnet_wrong = np.sum((y_pred_resnet == y_true_resnet) & (y_pred_effnet != y_true_effnet))
    effnet_correct_resnet_wrong = np.sum((y_pred_resnet != y_true_resnet) & (y_pred_effnet == y_true_effnet))
    
    print(f"\nMatriz de Concordância:")
    print(f"  Ambos corretos:        {both_correct}")
    print(f"  Ambos incorretos:      {both_wrong}")
    print(f"  ResNet correto, EffNet errado: {resnet_correct_effnet_wrong}")
    print(f"  EffNet correto, ResNet errado: {effnet_correct_resnet_wrong}")
    
    if resnet_correct_effnet_wrong + effnet_correct_resnet_wrong > 0:
        mcnemar_stat = (abs(resnet_correct_effnet_wrong - effnet_correct_resnet_wrong) - 1)**2 / \
                      (resnet_correct_effnet_wrong + effnet_correct_resnet_wrong)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        
        print(f"\nEstatística de McNemar: {mcnemar_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✓ Diferença estatisticamente significativa (p < 0.05)")
        else:
            print("✗ Diferença não estatisticamente significativa (p >= 0.05)")
    else:
        print("\n⚠ Não é possível realizar teste de McNemar (sem discordâncias)")
    
    print("\n" + "-" * 80)
    print("MÉTRICAS ADICIONAIS")
    print("-" * 80)
    
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
    
    kappa_resnet = metrics_resnet.get('kappa', cohen_kappa_score(y_true_resnet, y_pred_resnet))
    kappa_effnet = metrics_effnet.get('kappa', cohen_kappa_score(y_true_effnet, y_pred_effnet))
    
    mcc_resnet = metrics_resnet.get('mcc', matthews_corrcoef(y_true_resnet, y_pred_resnet))
    mcc_effnet = metrics_effnet.get('mcc', matthews_corrcoef(y_true_effnet, y_pred_effnet))
    
    print(f"\nCohen's Kappa:")
    print(f"  ResNet-50:       {kappa_resnet:.4f}")
    print(f"  EfficientNet-B0: {kappa_effnet:.4f}")
    print(f"  Diferença:       {kappa_effnet - kappa_resnet:+.4f}")
    
    print(f"\nMatthews Correlation Coefficient (MCC):")
    print(f"  ResNet-50:       {mcc_resnet:.4f}")
    print(f"  EfficientNet-B0: {mcc_effnet:.4f}")
    print(f"  Diferença:       {mcc_effnet - mcc_resnet:+.4f}")
    
    print("\n" + "="*60 + "\n")


def analyze_errors(y_true, y_pred_resnet, y_pred_effnet):
    """
    Analisa casos onde os modelos discordam ou cometem erros.
    """
    print("\n" + "="*60)
    print("ANÁLISE DE ERROS")
    print("="*60)
    
    resnet_correct_effnet_wrong = np.sum(
        (y_pred_resnet == y_true) & (y_pred_effnet != y_true)
    )
    
    effnet_correct_resnet_wrong = np.sum(
        (y_pred_effnet == y_true) & (y_pred_resnet != y_true)
    )
    
    fp_resnet = np.sum((y_pred_resnet == 1) & (y_true == 0))
    fn_resnet = np.sum((y_pred_resnet == 0) & (y_true == 1))
    
    fp_effnet = np.sum((y_pred_effnet == 1) & (y_true == 0))
    fn_effnet = np.sum((y_pred_effnet == 0) & (y_true == 1))
    
    print(f"\nDiscordâncias:")
    print(f"  ResNet acerta, EfficientNet erra: {resnet_correct_effnet_wrong}")
    print(f"  EfficientNet acerta, ResNet erra: {effnet_correct_resnet_wrong}")
    
    print(f"\nFalsos Positivos:")
    print(f"  ResNet-50:       {fp_resnet}")
    print(f"  EfficientNet-B0: {fp_effnet}")
    
    print(f"\nFalsos Negativos:")
    print(f"  ResNet-50:       {fn_resnet}")
    print(f"  EfficientNet-B0: {fn_effnet}")
    
    print("="*60 + "\n")


def main():
    """
    Função principal de comparação.
    """
    print("\n" + "="*60)
    print("COMPARAÇÃO COMPLETA: ResNet-50 vs EfficientNet-B0")
    print("="*60 + "\n")
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Configuração
    config = DEFAULT_CONFIG.copy()
    
    # Transformações e dados
    print("[1/5] Carregando dados...")
    val_transform = get_transforms(config, train=False)
    _, _, test_loader, _ = get_dataloaders(
        config, val_transform, val_transform
    )
    
    # ===== RESNET-50 =====
    print("\n[2/5] Avaliando ResNet-50...")
    print("-" * 40)
    
    resnet_model = get_resnet50(num_classes=2, pretrained=True).to(device)
    resnet_checkpoint = './checkpoints/resnet50/best_model.pth'
    resnet_model = load_trained_model(resnet_model, resnet_checkpoint, device)
    
    # Métricas clínicas
    metrics_resnet, _ = evaluate_model(resnet_model, test_loader, device)
    print_metrics(metrics_resnet, phase="ResNet-50 Test")
    
    # Eficiência computacional
    print("\nBenchmark de eficiência ResNet-50:")
    efficiency_resnet = benchmark_model(resnet_model, device, verbose=True)
    
    # Predições para ROC
    y_true_resnet, y_pred_resnet, y_proba_resnet = get_predictions(
        resnet_model, test_loader, device
    )
    
    # ===== EFFICIENTNET-B0 =====
    print("\n[3/5] Avaliando EfficientNet-B0...")
    print("-" * 40)
    
    effnet_model = get_efficientnet_b0(num_classes=2, pretrained=True).to(device)
    effnet_checkpoint = './checkpoints/efficientnet_b0/best_model.pth'
    effnet_model = load_trained_model(effnet_model, effnet_checkpoint, device)
    
    # Métricas clínicas
    metrics_effnet, _ = evaluate_model(effnet_model, test_loader, device)
    print_metrics(metrics_effnet, phase="EfficientNet-B0 Test")
    
    # Eficiência computacional
    print("\nBenchmark de eficiência EfficientNet-B0:")
    efficiency_effnet = benchmark_model(effnet_model, device, verbose=True)
    
    # Predições para ROC
    y_true_effnet, y_pred_effnet, y_proba_effnet = get_predictions(
        effnet_model, test_loader, device
    )
    
    # ===== COMPARAÇÕES =====
    print("\n[4/6] Gerando visualizações comparativas...")
    
    # Curvas ROC
    plot_roc_curves(
        y_true_resnet, y_proba_resnet, metrics_resnet['auc_roc'],
        y_true_effnet, y_proba_effnet, metrics_effnet['auc_roc']
    )
    
    # Comparação de métricas
    plot_metrics_comparison(metrics_resnet, metrics_effnet)
    
    # Comparação de eficiência
    print_efficiency_comparison(
        efficiency_resnet, efficiency_effnet,
        "ResNet-50", "EfficientNet-B0"
    )
    
    # ===== ANÁLISE ESTATÍSTICA =====
    print("\n[5/6] Análise estatística...")
    statistical_comparison(
        metrics_resnet, metrics_effnet,
        y_true_resnet, y_pred_resnet,
        y_true_effnet, y_pred_effnet
    )
    
    # ===== ANÁLISE DE ERROS =====
    print("\n[6/6] Análise de erros...")
    analyze_errors(
        y_true_resnet, y_pred_resnet, y_pred_effnet
    )
    
    # ===== RESUMO FINAL =====
    print("\n" + "="*60)
    print("RESUMO FINAL DA COMPARAÇÃO")
    print("="*60)
    
    print("\n📊 MÉTRICAS CLÍNICAS:")
    print(f"  ResNet-50       | AUC: {metrics_resnet['auc_roc']:.4f} | Sens: {metrics_resnet['sensitivity']:.4f}")
    print(f"  EfficientNet-B0 | AUC: {metrics_effnet['auc_roc']:.4f} | Sens: {metrics_effnet['sensitivity']:.4f}")
    
    if efficiency_resnet and efficiency_effnet:
        print("\n⚡ EFICIÊNCIA COMPUTACIONAL:")
        print(f"  ResNet-50       | {efficiency_resnet['flops']['gflops']:.2f} GFLOPs | {efficiency_resnet['latency']['mean_ms']:.2f} ms")
        print(f"  EfficientNet-B0 | {efficiency_effnet['flops']['gflops']:.2f} GFLOPs | {efficiency_effnet['latency']['mean_ms']:.2f} ms")
    
    print("\n✓ Comparação completa finalizada!")
    print("  Resultados salvos em: ./results/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()