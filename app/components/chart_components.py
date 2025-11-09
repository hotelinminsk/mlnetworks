"""
Chart Components - Modern & Interactive Visualizations
SOLID: Single Responsibility - Grafik oluşturma bileşenleri
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (
    COLOR_NORMAL, COLOR_ATTACK, TRAFFIC_THRESHOLD, MODERN_COLORS
)

# Modern chart templates
MODERN_LAYOUT = {
    'font': {'family': "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", 'size': 12, 'color': MODERN_COLORS['label_color']},
    'plot_bgcolor': MODERN_COLORS['plot_bg'],
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'hovermode': 'x unified',
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1,
        'bgcolor': 'rgba(255,255,255,0.8)',
        'bordercolor': 'rgba(0,0,0,0.1)',
        'borderwidth': 1,
        'font': {'size': 11}
    },
    'transition': {'duration': 500, 'easing': 'cubic-in-out'},
    'title': {
        'font': {'size': 20, 'family': "Inter, sans-serif", 'color': MODERN_COLORS['title_color']},
        'x': 0.5,
        'xanchor': 'center'
    },
    'xaxis': {
        'gridcolor': MODERN_COLORS['grid'],
        'gridwidth': 1,
        'showgrid': True,
        'zeroline': False,
        'title': {'font': {'size': 14}}
    },
    'yaxis': {
        'gridcolor': MODERN_COLORS['grid'],
        'gridwidth': 1,
        'showgrid': True,
        'zeroline': False,
        'title': {'font': {'size': 14}}
    }
}


class ChartComponents:
    """Grafik oluşturma bileşenleri"""
    
    @staticmethod
    def create_gauge_chart(
        value: float,
        threshold: float,
        title: str = "Saldırı Olasılığı"
    ) -> go.Figure:
        """Modern gauge chart oluştur"""
        # Risk seviyesine göre renk belirle
        if value >= threshold:
            bar_color = MODERN_COLORS['danger']
            step_colors = [
                {'range': [0, threshold*100], 'color': MODERN_COLORS['success']},
                {'range': [threshold*100, 100], 'color': MODERN_COLORS['danger']}
            ]
        else:
            bar_color = MODERN_COLORS['success']
            step_colors = [
                {'range': [0, threshold*100], 'color': MODERN_COLORS['success']},
                {'range': [threshold*100, 100], 'color': '#fee2e2'}
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': title,
                'font': {'size': 18, 'family': "Inter, sans-serif", 'color': MODERN_COLORS['title_color']}
            },
            delta={
                'reference': threshold * 100,
                'increasing': {'color': MODERN_COLORS['danger']},
                'decreasing': {'color': MODERN_COLORS['success']},
                'font': {'size': 14}
            },
            number={
                'font': {'size': 32, 'family': "Inter, sans-serif", 'color': MODERN_COLORS['title_color']},
                'suffix': '%'
            },
            gauge={
                'axis': {
                    'range': [None, 100],
                    'tickwidth': 1,
                    'tickcolor': MODERN_COLORS['label_color'],
                    'tickfont': {'size': 11, 'color': MODERN_COLORS['label_color']},
                    'nticks': 5
                },
                'bar': {
                    'color': bar_color,
                    'thickness': 0.3
                },
                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                'borderwidth': 2,
                'bordercolor': MODERN_COLORS['grid'],
                'steps': step_colors,
                'threshold': {
                    'line': {
                        'color': MODERN_COLORS['dark'],
                        'width': 3
                    },
                    'thickness': 0.8,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Inter, sans-serif"}
        )
        
        return fig
    
    @staticmethod
    def create_probability_distribution(
        proba_normal: np.ndarray,
        proba_attack: np.ndarray,
        threshold: float,
        title: str = "Olasılık Dağılımı"
    ) -> go.Figure:
        """Modern olasılık dağılım grafiği oluştur"""
        fig = go.Figure()
        
        # Normal trafik - gradient efektli
        fig.add_trace(go.Histogram(
            x=proba_normal,
            name='🟢 Normal Trafik',
            opacity=0.8,
            marker=dict(
                color=MODERN_COLORS['normal_traffic'],
                line=dict(color=MODERN_COLORS['light'], width=1)
            ),
            nbinsx=50,
            hovertemplate='<b>Normal Trafik</b><br>' +
                         'Olasılık: %{x:.3f}<br>' +
                         'Sayı: %{y}<extra></extra>'
        ))
        
        # Saldırı trafiği - gradient efektli
        fig.add_trace(go.Histogram(
            x=proba_attack,
            name='🔴 Saldırı Trafiği',
            opacity=0.8,
            marker=dict(
                color=MODERN_COLORS['attack_traffic'],
                line=dict(color=MODERN_COLORS['light'], width=1)
            ),
            nbinsx=50,
            hovertemplate='<b>Saldırı Trafiği</b><br>' +
                         'Olasılık: %{x:.3f}<br>' +
                         'Sayı: %{y}<extra></extra>'
        ))
        
        # Threshold çizgisi - modern stil
        fig.add_vline(
            x=threshold,
            line=dict(
                dash="dashdot",
                color=MODERN_COLORS['dark'],
                width=3
            ),
            annotation_text=f"<b>Eşik: {threshold:.3f}</b>",
            annotation=dict(
                font=dict(size=12, color=MODERN_COLORS['dark']),
                bgcolor=MODERN_COLORS['plot_bg'],
                bordercolor=MODERN_COLORS['grid'],
                borderwidth=1,
                borderpad=4
            ),
            annotation_position="top right"
        )
        
        # Modern layout
        fig.update_layout(
            title_text=title,
            xaxis_title_text="Saldırı Olasılığı",
            yaxis_title_text="Örnek Sayısı",
            barmode='overlay',
            height=450,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_traffic_monitoring_chart(
        timestamps: List[datetime],
        traffic: np.ndarray,
        attacks: np.ndarray
    ) -> go.Figure:
        """Modern trafik monitoring grafiği oluştur"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                '<b style="font-size:16px; color:#1e293b">📈 Network Traffic Over Time</b>',
                '<b style="font-size:16px; color:#1e293b">🚨 Attack Detection Timeline</b>'
            ),
            vertical_spacing=0.15,
            row_heights=[0.65, 0.35],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Traffic line - gradient fill ile
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=traffic,
                mode='lines',
                name='🌐 Traffic Volume',
                line=dict(
                    color=MODERN_COLORS['primary'],
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                fill='tozeroy',
                fillcolor=f"rgba(102, 126, 234, 0.2)",
                hovertemplate='<b>Traffic Volume</b><br>' +
                             'Time: %{x|%H:%M:%S}<br>' +
                             'Packets/sec: %{y:.0f}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Threshold line - modern stil
        fig.add_hline(
            y=TRAFFIC_THRESHOLD,
            line=dict(
                dash="dashdot",
                color=MODERN_COLORS['warning'],
                width=2.5
            ),
            annotation_text=f"<b>Threshold: {TRAFFIC_THRESHOLD}</b>",
            annotation=dict(
                font=dict(size=11, color=MODERN_COLORS['warning']),
                bgcolor=MODERN_COLORS['plot_bg'],
                bordercolor=MODERN_COLORS['warning'],
                borderwidth=1
            ),
            annotation_position="right",
            row=1, col=1
        )
        
        # Attack markers - modern stil
        attack_times = [timestamps[i] for i in range(len(attacks)) if attacks[i] == 1]
        attack_values = [1] * len(attack_times)
        
        if len(attack_times) > 0:
            fig.add_trace(
                go.Scatter(
                    x=attack_times,
                    y=attack_values,
                    mode='markers+text',
                    name='🚨 Attack Detected',
                    marker=dict(
                        color=MODERN_COLORS['danger'],
                        size=20,
                        symbol='x-thin',
                        line=dict(color=MODERN_COLORS['light'], width=3),
                        opacity=0.9
                    ),
                    text=['🚨'] * len(attack_times),
                    textposition="middle center",
                    textfont=dict(size=12, color=MODERN_COLORS['light']),
                    hovertemplate='<b>🚨 ATTACK DETECTED</b><br>' +
                                 'Time: %{x|%H:%M:%S}<br>' +
                                 'Severity: High<extra></extra>',
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # Axis updates - modern stil
        fig.update_xaxes(
            title_text="<b>Time</b>",
            gridcolor=MODERN_COLORS['grid'],
            gridwidth=1,
            showgrid=True,
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="<b>Packets/sec</b>",
            gridcolor=MODERN_COLORS['grid'],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="<b>Events</b>",
            range=[0, 1.5],
            gridcolor=MODERN_COLORS['grid'],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            row=2, col=1
        )
        
        # Modern layout
        layout_config = MODERN_LAYOUT.copy()
        layout_config.update({
            'height': 650,
            'margin': dict(l=50, r=30, t=60, b=50),
        })
        fig.update_layout(**layout_config)
        
        return fig
    
    @staticmethod
    def create_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        title: str = "ROC Curve"
    ) -> go.Figure:
        """Modern ROC curve grafiği oluştur"""
        fig = go.Figure()
        
        # ROC curve - gradient line
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC AUC = {auc:.3f}',
            line=dict(color=MODERN_COLORS['primary'], width=3.5, shape='spline'),
            fill='tozeroy',
            fillcolor=f"rgba(102, 126, 234, 0.2)",
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color=MODERN_COLORS['label_color'], width=2, dash='dash'),
            hovertemplate='Random Classifier<extra></extra>'
        ))
        
        # Modern layout
        fig.update_layout(
            title_text=title,
            xaxis_title_text="False Positive Rate",
            yaxis_title_text="True Positive Rate",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=450,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_heatmap(
        cm: np.ndarray,
        labels: List[str] = None,
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """Modern confusion matrix heatmap oluştur"""
        if labels is None:
            labels = ['Normal', 'Saldırı']
        
        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate='%{text}',
            colorscale='Viridis',
            hovertemplate='Gerçek: %{y}<br>Tahmin: %{x}<br>Sayı: %{z}<extra></extra>',
            colorbar=dict(
                title="Count",
                len=0.8,
                thickness=20,
                titlefont={'size': 12},
                tickfont={'size': 11}
            )
        ))
        
        # Modern layout
        fig.update_layout(
            title_text=title,
            xaxis_title_text="<b>Tahmin Edilen</b>",
            yaxis_title_text="<b>Gerçek</b>",
            height=450,
            **MODERN_LAYOUT
        )
        
        return fig
