"""
Real-Time Monitoring View
SOLID Principles:
- Single Responsibility: Sadece real-time monitoring tab'ının koordinasyonu
- Dependency Inversion: Service ve component'lere bağımlı
"""
import streamlit as st
import pandas as pd
from typing import Any

from services.realtime_monitoring_service import RealtimeMonitoringService
from components.realtime_chart import RealtimeChartComponent
from components.realtime_metrics import RealtimeMetricsComponent
from components.realtime_alerts import RealtimeAlertsComponent


class RealtimeMonitoringView:
    """Real-time monitoring view controller"""
    
    def __init__(self, max_points: int = 100):
        """
        Args:
            max_points: Kayan pencerede tutulacak maksimum nokta sayısı
        """
        self.service = RealtimeMonitoringService(max_points=max_points)
        self.chart_component = RealtimeChartComponent()
        self.metrics_component = RealtimeMetricsComponent()
        self.alerts_component = RealtimeAlertsComponent()
    
    def render_controls(self) -> None:
        """Control butonlarını render et"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            is_paused = st.toggle(
                "⏸️ Pause Monitoring",
                value=not self.service.is_active(),
                key="pause_monitoring"
            )
            
            if is_paused:
                self.service.pause()
            else:
                self.service.resume()
        
        with col2:
            if st.button("🔄 Reset", key="reset_monitoring"):
                self.service.reset()
    
    def render(self) -> None:
        """Ana render fonksiyonu"""
        # Controls
        self.render_controls()
        st.markdown("---")
        
        # Fragment'i çağır (her 1 saniyede otomatik çalışır)
        self._render_monitoring_fragment()
    
    @st.fragment(run_every="1s")
    def _render_monitoring_fragment(self) -> None:
        """Fragment: Sadece bu kısım yenilenir"""
        # Yeni veri noktası üret
        new_row = self.service.generate_data_point()
        
        if new_row is not None:
            self.service.update_data(new_row)
        
        # Mevcut veriyi al
        monitoring_df = self.service.get_current_data()
        
        if len(monitoring_df) == 0:
            st.info("📊 Monitoring starting... Data will appear in a moment.")
            return
        
        # Metrikleri hesapla
        metrics = self.service.calculate_metrics(monitoring_df)
        
        # Status indicator
        update_count = st.session_state.get('update_counter', 0)
        self.metrics_component.render_status(
            is_active=self.service.is_active(),
            update_count=update_count,
            data_points=len(monitoring_df)
        )
        
        # Metrics dashboard
        self.metrics_component.render_metrics(metrics)
        
        # Chart
        self.chart_component.render(monitoring_df)
        
        # Alerts
        self.alerts_component.render(monitoring_df)

