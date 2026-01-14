"""
Gera relatório completo em Markdown e HTML com todos os resultados.

Uso:
    python experiments/generate_report.py
"""

import sys
sys.path.append('.')

import json
from pathlib import Path
from datetime import datetime


def load_results(results_path='./results/results.json'):
    """Carrega resultados do JSON."""
    if not Path(results_path).exists():
        print(f"❌ Arquivo de resultados não encontrado: {results_path}")
        print("   Execute primeiro: python experiments/main.py")
        sys.exit(1)
    
    with open(results_path, 'r') as f:
        return json.load(f)


def format_metrics_table(metrics, model_name):
    """Formata tabela de métricas."""
    rows = [
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| **Acurácia** | {metrics['accuracy']:.4f} |",
        f"| **Sensibilidade** | {metrics['sensitivity']:.4f} |",
        f"| **Especificidade** | {metrics['specificity']:.4f} |",
        f"| **Precisão** | {metrics['precision']:.4f} |",
        f"| **F1-Score** | {metrics['f1_score']:.4f} |",
        f"| **AUC-ROC** | {metrics['auc_roc']:.4f} |",
    ]
    
    if 'kappa' in metrics:
        rows.append(f"| **Cohen's Kappa** | {metrics['kappa']:.4f} |")
    if 'mcc' in metrics:
        rows.append(f"| **MCC** | {metrics['mcc']:.4f} |")
    
    return '\n'.join(rows)


def format_confusion_matrix(cm):
    """Formata matriz de confusão."""
    return f"""
```
                Predito
            Benigno  Maligno
Real Benigno   {cm[0][0]:5d}    {cm[0][1]:5d}
     Maligno   {cm[1][0]:5d}    {cm[1][1]:5d}
```
"""


def format_efficiency_table(efficiency):
    """Formata tabela de eficiência."""
    if not efficiency:
        return "| Métrica | Valor |\n|---------|-------|\n| N/A | Dados não disponíveis |"
    
    rows = [
        "| Métrica | Valor |",
        "|---------|-------|",
    ]
    
    if 'flops' in efficiency:
        rows.append(f"| **FLOPs** | {efficiency['flops']['gflops']:.2f} GFLOPs |")
    if 'latency' in efficiency:
        rows.append(f"| **Latência Média** | {efficiency['latency']['mean_ms']:.2f} ms |")
        rows.append(f"| **Latência Std** | {efficiency['latency']['std_ms']:.2f} ms |")
    if 'memory' in efficiency:
        rows.append(f"| **Memória GPU** | {efficiency['memory']['peak_mb']:.2f} MB |")
    if 'size' in efficiency:
        rows.append(f"| **Tamanho do Modelo** | {efficiency['size']['size_mb']:.2f} MB |")
    
    return '\n'.join(rows)


def generate_markdown_report(results):
    """Gera relatório em Markdown."""
    
    md = f"""# Relatório de Análise Comparativa: ResNet-50 vs EfficientNet-B0
## Detecção de Melanoma

**Data de Geração:** {results['timestamp']}

---

## 📋 Sumário Executivo

Este relatório apresenta uma análise comparativa completa entre dois modelos de deep learning 
para detecção de melanoma: ResNet-50 e EfficientNet-B0. Os modelos foram treinados no dataset 
ISIC 2020 e avaliados usando métricas clínicas e de eficiência computacional.

---

## 🎯 Resultados do Treinamento

### ResNet-50

#### Métricas de Desempenho

{format_metrics_table(results['resnet50']['test_metrics'], 'ResNet-50')}

#### Matriz de Confusão

{format_confusion_matrix(results['resnet50']['confusion_matrix'])}

#### Eficiência Computacional

{format_efficiency_table(results['resnet50'].get('efficiency', {}))}

#### Informações do Modelo

- **Parâmetros Totais:** {results['resnet50']['num_parameters']:,}
- **Melhor AUC-ROC (Validação):** {results['resnet50']['best_val_auc']:.4f}
- **AUC-ROC (Teste):** {results['resnet50']['test_metrics']['auc_roc']:.4f}

---

### EfficientNet-B0

#### Métricas de Desempenho

{format_metrics_table(results['efficientnet_b0']['test_metrics'], 'EfficientNet-B0')}

#### Matriz de Confusão

{format_confusion_matrix(results['efficientnet_b0']['confusion_matrix'])}

#### Eficiência Computacional

{format_efficiency_table(results['efficientnet_b0'].get('efficiency', {}))}

#### Informações do Modelo

- **Parâmetros Totais:** {results['efficientnet_b0']['num_parameters']:,}
- **Melhor AUC-ROC (Validação):** {results['efficientnet_b0']['best_val_auc']:.4f}
- **AUC-ROC (Teste):** {results['efficientnet_b0']['test_metrics']['auc_roc']:.4f}

---

## 📊 Comparação Direta

### Métricas Clínicas

| Métrica | ResNet-50 | EfficientNet-B0 | Diferença |
|---------|-----------|-----------------|-----------|
| **AUC-ROC** | {results['resnet50']['test_metrics']['auc_roc']:.4f} | {results['efficientnet_b0']['test_metrics']['auc_roc']:.4f} | {results['efficientnet_b0']['test_metrics']['auc_roc'] - results['resnet50']['test_metrics']['auc_roc']:+.4f} |
| **Acurácia** | {results['resnet50']['test_metrics']['accuracy']:.4f} | {results['efficientnet_b0']['test_metrics']['accuracy']:.4f} | {results['efficientnet_b0']['test_metrics']['accuracy'] - results['resnet50']['test_metrics']['accuracy']:+.4f} |
| **Sensibilidade** | {results['resnet50']['test_metrics']['sensitivity']:.4f} | {results['efficientnet_b0']['test_metrics']['sensitivity']:.4f} | {results['efficientnet_b0']['test_metrics']['sensitivity'] - results['resnet50']['test_metrics']['sensitivity']:+.4f} |
| **Especificidade** | {results['resnet50']['test_metrics']['specificity']:.4f} | {results['efficientnet_b0']['test_metrics']['specificity']:.4f} | {results['efficientnet_b0']['test_metrics']['specificity'] - results['resnet50']['test_metrics']['specificity']:+.4f} |
| **Precisão** | {results['resnet50']['test_metrics']['precision']:.4f} | {results['efficientnet_b0']['test_metrics']['precision']:.4f} | {results['efficientnet_b0']['test_metrics']['precision'] - results['resnet50']['test_metrics']['precision']:+.4f} |
| **F1-Score** | {results['resnet50']['test_metrics']['f1_score']:.4f} | {results['efficientnet_b0']['test_metrics']['f1_score']:.4f} | {results['efficientnet_b0']['test_metrics']['f1_score'] - results['resnet50']['test_metrics']['f1_score']:+.4f} |

### Eficiência Computacional

"""
    
    # Adicionar comparação de eficiência se disponível
    resnet_eff = results['resnet50'].get('efficiency', {})
    effnet_eff = results['efficientnet_b0'].get('efficiency', {})
    
    if resnet_eff and effnet_eff:
        md += """| Métrica | ResNet-50 | EfficientNet-B0 | Diferença |
|---------|-----------|-----------------|-----------|
"""
        
        if 'flops' in resnet_eff and 'flops' in effnet_eff:
            diff = effnet_eff['flops']['gflops'] - resnet_eff['flops']['gflops']
            md += f"| **FLOPs** | {resnet_eff['flops']['gflops']:.2f} GFLOPs | {effnet_eff['flops']['gflops']:.2f} GFLOPs | {diff:+.2f} GFLOPs |\n"
        
        if 'latency' in resnet_eff and 'latency' in effnet_eff:
            diff = effnet_eff['latency']['mean_ms'] - resnet_eff['latency']['mean_ms']
            md += f"| **Latência** | {resnet_eff['latency']['mean_ms']:.2f} ms | {effnet_eff['latency']['mean_ms']:.2f} ms | {diff:+.2f} ms |\n"
    
    md += f"""
---

## 📈 Análise de Curvas de Treinamento

Os gráficos de treinamento estão disponíveis no TensorBoard:
- ResNet-50: `./runs/resnet50/`
- EfficientNet-B0: `./runs/efficientnet_b0/`

Visualize com: `tensorboard --logdir ./runs`

---

## 🔍 Conclusões

### ResNet-50
- **Pontos Fortes:** {'Modelo robusto e amplamente testado' if results['resnet50']['test_metrics']['auc_roc'] > 0.8 else 'Modelo com desempenho moderado'}
- **Pontos Fracos:** {'Modelo mais pesado computacionalmente' if resnet_eff.get('flops', {}).get('gflops', 0) > 4 else 'Modelo relativamente eficiente'}

### EfficientNet-B0
- **Pontos Fortes:** {'Modelo eficiente com boa relação desempenho/complexidade' if effnet_eff.get('flops', {}).get('gflops', 0) < 1 else 'Modelo com boa capacidade de aprendizado'}
- **Pontos Fracos:** {'Pode requerer mais ajustes de hiperparâmetros' if results['efficientnet_b0']['test_metrics']['auc_roc'] < results['resnet50']['test_metrics']['auc_roc'] else 'Desempenho competitivo'}

### Recomendação

"""
    
    # Determinar melhor modelo
    resnet_auc = results['resnet50']['test_metrics']['auc_roc']
    effnet_auc = results['efficientnet_b0']['test_metrics']['auc_roc']
    
    if abs(resnet_auc - effnet_auc) < 0.01:
        md += "Ambos os modelos apresentam desempenho similar. A escolha deve ser baseada em requisitos específicos:\n"
        md += "- **ResNet-50**: Melhor para aplicações que priorizam robustez e interpretabilidade\n"
        md += "- **EfficientNet-B0**: Melhor para aplicações que priorizam eficiência computacional\n"
    elif resnet_auc > effnet_auc:
        md += f"**ResNet-50** apresenta melhor desempenho (AUC-ROC: {resnet_auc:.4f} vs {effnet_auc:.4f}) e é recomendado para aplicações que priorizam acurácia máxima.\n"
    else:
        md += f"**EfficientNet-B0** apresenta melhor desempenho (AUC-ROC: {effnet_auc:.4f} vs {resnet_auc:.4f}) e é recomendado por combinar boa acurácia com eficiência computacional.\n"
    
    md += f"""
---

## 📁 Arquivos Gerados

- **Checkpoints:** `./checkpoints/`
- **Logs TensorBoard:** `./runs/`
- **Visualizações:** `./results/`
- **Dados de Resultados:** `./results/results.json`

---

**Relatório gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return md


def save_report(markdown_content, output_dir='./results'):
    """Salva relatório em Markdown e HTML."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar Markdown
    md_path = output_path / 'relatorio_completo.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"✓ Relatório Markdown salvo em: {md_path}")
    
    # Converter para HTML (requer markdown)
    try:
        import markdown
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relatório - Detecção de Melanoma</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
        
        html_path = output_path / 'relatorio_completo.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"✓ Relatório HTML salvo em: {html_path}")
    except ImportError:
        print("⚠ Biblioteca 'markdown' não instalada. Instale com: pip install markdown")
        print("  Apenas versão Markdown foi gerada.")


def main():
    """Função principal."""
    print("\n" + "="*80)
    print("GERAÇÃO DE RELATÓRIO COMPLETO")
    print("="*80 + "\n")
    
    # Carregar resultados
    results = load_results('./results/results.json')
    
    # Gerar relatório
    print("Gerando relatório...")
    markdown_report = generate_markdown_report(results)
    
    # Salvar
    save_report(markdown_report)
    
    print("\n" + "="*80)
    print("RELATÓRIO GERADO COM SUCESSO!")
    print("="*80)
    print("\nArquivos gerados:")
    print("  - ./results/relatorio_completo.md")
    print("  - ./results/relatorio_completo.html (se markdown instalado)")
    print("\nAbra o arquivo HTML no navegador para visualização completa.")


if __name__ == '__main__':
    main()

