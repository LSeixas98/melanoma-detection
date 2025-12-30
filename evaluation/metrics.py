"""
Módulo para cálculo de métricas de avaliação.
"""

from typing import Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef
)


def evaluate_model(model, dataloader, device, criterion: Optional[torch.nn.Module] = None) -> Tuple[Dict, Optional[float]]:
    """
    Avalia modelo e retorna métricas clínicas completas.
    
    Calcula: acurácia, sensibilidade, especificidade, precisão, F1-Score, AUC-ROC
    e matriz de confusão.
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'kappa': kappa,
        'mcc': mcc,
        'confusion_matrix': cm
    }
    
    avg_loss = running_loss / len(dataloader.dataset) if criterion is not None else None
    
    return metrics, avg_loss


def print_metrics(metrics: Dict, phase: str = "") -> None:
    """Imprime métricas formatadas para visualização no console."""
    print(f"\n{'='*60}")
    if phase:
        print(f"MÉTRICAS - {phase.upper()}")
    else:
        print("MÉTRICAS")
    print(f"{'='*60}")
    
    print(f"  Acurácia:      {metrics['accuracy']:.4f}")
    print(f"  Sensibilidade: {metrics['sensitivity']:.4f}")
    print(f"  Especificidade: {metrics['specificity']:.4f}")
    print(f"  Precisão:      {metrics['precision']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
    
    if 'kappa' in metrics:
        print(f"  Cohen's Kappa: {metrics['kappa']:.4f}")
    if 'mcc' in metrics:
        print(f"  MCC:           {metrics['mcc']:.4f}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\n  Matriz de Confusão:")
        print(f"    [{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"    [{cm[1,0]:5d}  {cm[1,1]:5d}]")
    
    print(f"{'='*60}\n")


def get_predictions(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtém predições completas do modelo.
    
    Returns:
        Tupla (y_true, y_pred, y_proba) onde y_proba é a probabilidade da classe positiva
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
