"""
Real-Time Metrics Component
SOLID Principles:
- Single Responsibility: Sadece metrik görselleştirmesi
"""
import pandas as pd
import streamlit as st
from typing import Dict


class RealtimeMetricsComponent:
    """Real-time monitoring metrikleri için component"""
    
    def render_status(self, is_active: bool, update_count: int, data_points: int) -> None:
        """
        Status indicator render et
        
        Args:
            is_active: Monitoring aktif mi?
            update_count: Güncelleme sayısı
            data_points: Veri noktası sayısı
        """
        status_text = "🟢 LIVE" if is_active else "⏸️ PAUSED"
        
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:10px;'
            f'background:rgba(16,185,129,0.15);border-radius:8px;margin-bottom:15px;'
            f'border-left:4px solid #10b981;">'
            f'<span style="color:#10b981;font-weight:700;font-size:14px;">{status_text}</span>'
            f'<span style="color:#64748b;margin-left:auto;font-size:11px;">'
            f'Updates: {update_count} | Points: {data_points}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    def render_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Metrik kartlarını render et
        
        Args:
            metrics: Metrikler dictionary
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Trafik", f"{int(metrics['total_traffic']):,}")
        
        with col2:
            st.metric("Saldırı Sayısı", int(metrics['attack_count']))
        
        with col3:
            st.metric("Saldırı Oranı", f"{metrics['attack_rate']:.1f}%")
        
        with col4:
            st.metric("Ortalama Trafik", f"{int(metrics['avg_traffic'])}")


