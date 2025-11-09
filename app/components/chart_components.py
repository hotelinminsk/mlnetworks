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

from app.config import COLOR_NORMAL, COLOR_ATTACK, TRAFFIC_THRESHOLD

# Modern color palettes
MODERN_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2',
    'normal_traffic': '#10b981',
    'attack_traffic': '#ef4444',
    'background': '#f8fafc',
    'grid': '#e2e8f0'
}

# Modern chart templates
MODERN_LAYOUT = {
    'font': {'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', 'size': 12},
    'plot_bgcolor': 'rgba(0,0,0,0)',
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
        'borderwidth': 1
    },
    'xaxis': {
        'gridcolor': MODERN_COLORS['grid'],
        'gridwidth': 1,
        'showgrid': True,
        'zeroline': False
    },
    'yaxis': {
        'gridcolor': MODERN_COLORS['grid'],
        'gridwidth': 1,
        'showgrid': True,
        'zeroline': False
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
                'font': {'size': 18, 'family': 'Inter, sans-serif', 'color': '#1e293b'}
            },
            delta={
                'reference': threshold * 100,
                'increasing': {'color': MODERN_COLORS['danger']},
                'decreasing': {'color': MODERN_COLORS['success']},
                'font': {'size': 14}
            },
            number={
                'font': {'size': 32, 'family': 'Inter, sans-serif', 'color': '#1e293b'},
                'suffix': '%'
            },
            gauge={
                'axis': {
                    'range': [None, 100],
                    'tickwidth': 1,
                    'tickcolor': '#64748b',
                    'tickfont': {'size': 11, 'color': '#64748b'},
                    'nticks': 5
                },
                'bar': {
                    'color': bar_color,
                    'thickness': 0.3
                },
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': '#e2e8f0',
                'steps': step_colors,
                'threshold': {
                    'line': {
                        'color': '#1e293b',
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
            plot_bgcolor='rgba(0,0,0,0)'
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
            opacity=0.75,
            marker=dict(
                color=MODERN_COLORS['normal_traffic'],
                line=dict(color='white', width=1)
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
            opacity=0.75,
            marker=dict(
                color=MODERN_COLORS['attack_traffic'],
                line=dict(color='white', width=1)
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
                color='#1e293b',
                width=3
            ),
            annotation_text=f"<b>Eşik: {threshold:.3f}</b>",
            annotation=dict(
                font=dict(size=12, color='#1e293b'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e2e8f0',
                borderwidth=1,
                borderpad=4
            )
        )
        
        # Modern layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Inter, sans-serif', color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Saldırı Olasılığı",
                titlefont=dict(size=14, color='#64748b'),
                gridcolor=MODERN_COLORS['grid'],
                gridwidth=1,
                zeroline=False
            ),
            yaxis=dict(
                title="Örnek Sayısı",
                titlefont=dict(size=14, color='#64748b'),
                gridcolor=MODERN_COLORS['grid'],
                gridwidth=1,
                zeroline=False
            ),
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
                fillcolor=f"rgba(102, 126, 234, 0.15)",
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
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=MODERN_COLORS['warning'],
                borderwidth=1
            ),
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
                        size=18,
                        symbol='x',
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    text=['⚠️'] * len(attack_times),
                    textposition="middle center",
                    textfont=dict(size=10),
                    hovertemplate='<b>🚨 ATTACK DETECTED</b><br>' +
                                 'Time: %{x|%H:%M:%S}<br>' +
                                 'Severity: High<extra></extra>',
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # Axis updates - modern stil
        fig.update_xaxes(
            title="<b>Time</b>",
            title_font=dict(size=13, color='#64748b'),
            gridcolor=MODERN_COLORS['grid'],
            gridwidth=1,
            showgrid=True,
            row=2, col=1
        )
        
        fig.update_yaxes(
            title="<b>Packets/sec</b>",
            title_font=dict(size=13, color='#64748b'),
            gridcolor=MODERN_COLORS['grid'],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            row=1, col=1
        )
        
        fig.update_yaxes(
            title="<b>Events</b>",
            title_font=dict(size=13, color='#64748b'),
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
            'hovermode': 'x unified'
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
            name=f'ROC Curve (AUC = {auc:.3f})',
            line=dict(
                color=MODERN_COLORS['primary'],
                width=3,
                shape='spline',
                smoothing=1.0
            ),
            fill='tozeroy',
            fillcolor=f"rgba(102, 126, 234, 0.1)",
            hovertemplate='<b>ROC Curve</b><br>' +
                         'FPR: %{x:.3f}<br>' +
                         'TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(
                color='#94a3b8',
                width=2,
                dash='dash'
            ),
            hovertemplate='Random Classifier<extra></extra>'
        ))
        
        # Modern layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Inter, sans-serif', color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="False Positive Rate",
                titlefont=dict(size=14, color='#64748b'),
                range=[0, 1],
                gridcolor=MODERN_COLORS['grid'],
                gridwidth=1
            ),
            yaxis=dict(
                title="True Positive Rate",
                titlefont=dict(size=14, color='#64748b'),
                range=[0, 1],
                gridcolor=MODERN_COLORS['grid'],
                gridwidth=1
            ),
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
            colorscale=[
                [0, '#f1f5f9'],
                [0.5, MODERN_COLORS['info']],
                [1, MODERN_COLORS['primary']]
            ],
            text=cm,
            texttemplate='%{text}',
            textfont=dict(size=16, color='white', family='Inter, sans-serif'),
            hovertemplate='<b>%{y} → %{x}</b><br>' +
                         'Değer: %{z}<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Count",
                    font=dict(size=12, color='#64748b')
                ),
                tickfont=dict(size=11, color='#64748b')
            )
        ))
        
        # Modern layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Inter, sans-serif', color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="<b>Tahmin Edilen</b>",
                titlefont=dict(size=14, color='#64748b'),
                gridcolor=MODERN_COLORS['grid']
            ),
            yaxis=dict(
                title="<b>Gerçek</b>",
                titlefont=dict(size=14, color='#64748b'),
                gridcolor=MODERN_COLORS['grid']
            ),
            height=450,
            **MODERN_LAYOUT
        )
        
        return fig
