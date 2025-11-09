"""
Real-Time Chart Component
SOLID Principles:
- Single Responsibility: Sadece chart görselleştirmesi
- Open/Closed: Yeni chart tipleri eklenebilir
"""
import pandas as pd
import altair as alt
import streamlit as st
from typing import Optional


class RealtimeChartComponent:
    """Real-time monitoring chart'ı için component"""
    
    def __init__(self, height: int = 400, max_traffic: int = 500):
        """
        Args:
            height: Chart yüksekliği
            max_traffic: Y-axis maksimum değeri
        """
        self.height = height
        self.max_traffic = max_traffic
    
    def render(self, df: pd.DataFrame) -> None:
        """
        Altair chart'ı render et
        
        Args:
            df: Monitoring DataFrame
        """
        if len(df) == 0:
            st.info("📊 Monitoring starting... Data will appear in a moment.")
            return
        
        # Prepare data for Altair
        chart_df = df.copy()
        chart_df['time_str'] = chart_df['timestamp'].dt.strftime('%H:%M:%S')
        
        # Create chart
        chart = self._create_chart(chart_df)
        
        # Render
        st.altair_chart(chart, use_container_width=True)
    
    def _create_chart(self, df: pd.DataFrame) -> alt.Chart:
        """
        Altair chart oluştur
        
        Args:
            df: Hazırlanmış DataFrame
            
        Returns:
            Altair Chart objesi
        """
        # Base chart
        base = alt.Chart(df).encode(
            x=alt.X('timestamp:T', title='Time', axis=alt.Axis(format='%H:%M:%S'))
        )
        
        # Traffic line
        traffic_line = base.mark_line(color='#667eea', size=3).encode(
            y=alt.Y(
                'total_traffic:Q',
                title='Traffic (packets/s)',
                scale=alt.Scale(domain=[0, self.max_traffic])
            ),
            tooltip=[
                alt.Tooltip('time_str:N', title='Time'),
                alt.Tooltip('total_traffic:Q', title='Total Traffic', format='.1f'),
                alt.Tooltip('is_attack:Q', title='Attack')
            ]
        )
        
        # Attack markers (red dots)
        attack_points = base.transform_filter(
            alt.datum.is_attack == 1
        ).mark_circle(size=100, color='#ef4444', opacity=0.8).encode(
            y='total_traffic:Q',
            tooltip=[
                alt.Tooltip('time_str:N', title='Time'),
                alt.Tooltip('total_traffic:Q', title='Traffic', format='.1f')
            ]
        )
        
        # Combine layers
        chart = (traffic_line + attack_points).properties(
            width='container',
            height=self.height,
            title='Real-Time Network Traffic Monitoring'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=18,
            anchor='start'
        )
        
        return chart


