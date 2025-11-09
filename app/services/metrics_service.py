"""
Metrics Service Layer
SOLID: Single Responsibility - Metrik hesaplama işlemleri
"""
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class MetricsService:
    """Metrik hesaplama işlemleri için service sınıfı"""
    
    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Binary sınıflandırma metriklerini hesapla
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            
        Returns:
            Dict: Metrik isimleri ve değerleri
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    
    @staticmethod
    def calculate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Confusion matrix değerlerini hesapla
        
        Returns:
            Tuple: (TN, FP, FN, TP)
        """
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            return tn, fp, fn, tp
        else:
            # Edge case: single class
            return 0, 0, 0, int(cm[0, 0])
    
    @staticmethod
    def calculate_roc_auc(
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """ROC AUC skorunu hesapla"""
        return roc_auc_score(y_true, y_scores)
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Tüm metrikleri hesapla
        
        Returns:
            Dict: Tüm metrikler
        """
        metrics = MetricsService.calculate_binary_metrics(y_true, y_pred)
        metrics['roc_auc'] = MetricsService.calculate_roc_auc(y_true, y_scores)
        
        tn, fp, fn, tp = MetricsService.calculate_confusion_matrix(y_true, y_pred)
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        return metrics

