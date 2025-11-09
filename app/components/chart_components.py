"""
Chart Components
SOLID: Single Responsibility - Grafik oluşturma bileşenleri
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional
from datetime import datetime

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import COLOR_NORMAL, COLOR_ATTACK, TRAFFIC_THRESHOLD


class ChartComponents:
    """Grafik oluşturma bileşenleri"""
    
    @staticmethod
    def create_gauge_chart(
        value: float,
        threshold: float,
        title: str = "Saldırı Olasılığı"
    ) -> go.Figure:
        """Gauge chart oluştur"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': threshold * 100},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if value >= threshold else "green"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "lightgreen"},
                    {'range': [threshold*100, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def create_probability_distribution(
        proba_normal: np.ndarray,
        proba_attack: np.ndarray,
        threshold: float,
        title: str = "Olasılık Dağılımı"
    ) -> go.Figure:
        """Olasılık dağılım grafiği oluştur"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=proba_normal,
            name='Normal',
            opacity=0.7,
            marker_color=COLOR_NORMAL
        ))
        
        fig.add_trace(go.Histogram(
            x=proba_attack,
            name='Saldırı',
            opacity=0.7,
            marker_color=COLOR_ATTACK
        ))
        
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Threshold: {threshold}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Saldırı Olasılığı",
            yaxis_title="Sayı",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_traffic_monitoring_chart(
        timestamps: List[datetime],
        traffic: np.ndarray,
        attacks: np.ndarray
    ) -> go.Figure:
        """Trafik monitoring grafiği oluştur"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📈 Network Traffic Over Time', '🚨 Attack Detection Timeline'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Traffic line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=traffic,
                mode='lines',
                name='Traffic Volume',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.3)'
            ),
            row=1, col=1
        )
        
        # Threshold line
        fig.add_hline(
            y=TRAFFIC_THRESHOLD,
            line_dash="dash",
            line_color="orange",
            annotation_text="Threshold",
            row=1, col=1
        )
        
        # Attack markers
        attack_times = [timestamps[i] for i in range(len(attacks)) if attacks[i] == 1]
        attack_values = [1] * len(attack_times)
        
        fig.add_trace(
            go.Scatter(
                x=attack_times,
                y=attack_values,
                mode='markers',
                name='Attack Detected',
                marker=dict(
                    color='red',
                    size=15,
                    symbol='x',
                    line=dict(color='darkred', width=2)
                )
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Packets/sec", row=1, col=1)
        fig.update_yaxes(title_text="Events", row=2, col=1, range=[0, 2])
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

