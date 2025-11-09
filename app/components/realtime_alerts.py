"""
Real-Time Alerts Component
SOLID Principles:
- Single Responsibility: Sadece alert görselleştirmesi
"""
import pandas as pd
import streamlit as st


class RealtimeAlertsComponent:
    """Real-time monitoring alertleri için component"""
    
    def __init__(self, max_alerts: int = 5):
        """
        Args:
            max_alerts: Gösterilecek maksimum alert sayısı
        """
        self.max_alerts = max_alerts
    
    def render(self, df: pd.DataFrame) -> None:
        """
        Alert panelini render et
        
        Args:
            df: Monitoring DataFrame
        """
        # Son saldırıları filtrele
        recent_attacks = df[df['is_attack'] == 1].tail(self.max_alerts)
        
        if len(recent_attacks) == 0:
            return
        
        st.markdown("### 🚨 Recent Attacks")
        
        for idx, row in recent_attacks.iterrows():
            st.error(
                f"**ALERT** | Time: {row['timestamp'].strftime('%H:%M:%S')} | "
                f"Traffic: {int(row['total_traffic'])} pkt/s"
            )


