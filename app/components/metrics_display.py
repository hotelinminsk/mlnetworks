"""
Metrics Display Components
SOLID: Single Responsibility - Metrik görüntüleme bileşenleri
"""
import streamlit as st
import numpy as np
from typing import Dict


class MetricsDisplay:
    """Metrik görüntüleme bileşenleri"""
    
    @staticmethod
    def display_performance_metrics(metrics: Dict[str, float]) -> None:
        """Ana performans metriklerini göster"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with col2:
            st.metric("🔍 Precision", f"{metrics['precision']*100:.2f}%")
        with col3:
            st.metric("📡 Recall", f"{metrics['recall']*100:.2f}%")
        with col4:
            st.metric("⚖️ F1 Score", f"{metrics['f1']*100:.2f}%")
    
    @staticmethod
    def display_confusion_matrix_metrics(tn: int, fp: int, fn: int, tp: int) -> None:
        """Confusion matrix metriklerini göster"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ True Negative", f"{tn:,}")
        with col2:
            st.metric("❌ False Positive", f"{fp:,}")
        with col3:
            st.metric("❌ False Negative", f"{fn:,}")
        with col4:
            st.metric("✅ True Positive", f"{tp:,}")
    
    @staticmethod
    def display_summary_stats(y_test: np.ndarray) -> None:
        """Özet istatistikleri göster"""
        attack_count = int(y_test.sum())
        normal_count = int(len(y_test) - y_test.sum())
        attack_ratio = (attack_count / len(y_test)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Saldırı", f"{attack_count:,}")
            st.metric("🟢 Normal", f"{normal_count:,}")
        with col2:
            st.metric("📊 Saldırı %", f"{attack_ratio:.1f}%")
            st.metric("⚖️ Denge", f"{100-attack_ratio:.1f}%")

