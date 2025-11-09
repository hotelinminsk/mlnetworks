"""
Chart Components - Premium Modern Visualizations
Professional-grade charts with modern UI/UX principles
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
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

# Premium Modern Layout Template
MODERN_LAYOUT = {
    'font': {
        'family': "Inter, -apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        'size': 13,
        'color': MODERN_COLORS['dark']
    },
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
        'bgcolor': 'rgba(255, 255, 255, 0.95)',
        'bordercolor': MODERN_COLORS['grid'],
        'borderwidth': 1,
        'font': {'size': 12, 'color': MODERN_COLORS['dark']}
    },
    'transition': {
        'duration': 750,
        'easing': 'cubic-in-out'
    },
    'hoverlabel': {
        'bgcolor': 'rgba(255, 255, 255, 0.98)',
        'bordercolor': MODERN_COLORS['primary'],
        'font': {'size': 13, 'family': 'Inter', 'color': MODERN_COLORS['dark']}
    },
    'xaxis': {
        'gridcolor': 'rgba(203, 213, 225, 0.3)',
        'gridwidth': 0.5,
        'showgrid': True,
        'zeroline': False
    },
    'yaxis': {
        'gridcolor': 'rgba(203, 213, 225, 0.3)',
        'gridwidth': 0.5,
        'showgrid': True,
        'zeroline': False
    }
}


class ChartComponents:
    """Premium chart components with modern aesthetics"""
    
    @staticmethod
    def create_gauge_chart(
        value: float,
        threshold: float,
        title: str = "Attack Probability"
    ) -> go.Figure:
        """Premium gauge chart with gradient effects"""
        # Determine color based on risk level
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
                {'range': [threshold*100, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 20, 'family': 'Inter', 'color': MODERN_COLORS['dark']}
            },
            delta={
                'reference': threshold * 100,
                'increasing': {'color': MODERN_COLORS['danger']},
                'decreasing': {'color': MODERN_COLORS['success']},
                'font': {'size': 16}
            },
            number={
                'font': {'size': 42, 'family': 'Inter', 'color': MODERN_COLORS['dark']},
                'suffix': '%',
                'valueformat': '.2f'
            },
            gauge={
                'axis': {
                    'range': [None, 100],
                    'tickwidth': 2,
                    'tickcolor': MODERN_COLORS['gray_light'],
                    'tickfont': {'size': 12, 'color': MODERN_COLORS['gray']},
                    'nticks': 6
                },
                'bar': {
                    'color': bar_color,
                    'thickness': 0.35,
                    'line': {'color': 'white', 'width': 2}
                },
                'bgcolor': 'rgba(255, 255, 255, 0.95)',
                'borderwidth': 3,
                'bordercolor': MODERN_COLORS['grid'],
                'steps': step_colors,
                'threshold': {
                    'line': {
                        'color': MODERN_COLORS['dark'],
                        'width': 4
                    },
                    'thickness': 0.85,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(
            height=380,
            margin=dict(l=30, r=30, t=60, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter'}
        )
        
        return fig
    
    @staticmethod
    def create_probability_distribution(
        proba_normal: np.ndarray,
        proba_attack: np.ndarray,
        threshold: float,
        title: str = "Probability Distribution"
    ) -> go.Figure:
        """Premium probability distribution with modern styling"""
        fig = go.Figure()
        
        # Normal traffic - gradient effect
        fig.add_trace(go.Histogram(
            x=proba_normal,
            name='Normal Traffic',
            opacity=0.85,
            marker=dict(
                color=MODERN_COLORS['success'],
                line=dict(color='white', width=2),
                pattern=dict(shape="")
            ),
            nbinsx=50,
            hovertemplate='<b>Normal Traffic</b><br>' +
                         'Probability: %{x:.3f}<br>' +
                         'Count: %{y}<extra></extra>'
        ))
        
        # Attack traffic - gradient effect
        fig.add_trace(go.Histogram(
            x=proba_attack,
            name='Attack Traffic',
            opacity=0.85,
            marker=dict(
                color=MODERN_COLORS['danger'],
                line=dict(color='white', width=2),
                pattern=dict(shape="")
            ),
            nbinsx=50,
            hovertemplate='<b>Attack Traffic</b><br>' +
                         'Probability: %{x:.3f}<br>' +
                         'Count: %{y}<extra></extra>'
        ))
        
        # Threshold line with annotation
        fig.add_vline(
            x=threshold,
            line=dict(
                dash="dashdot",
                color=MODERN_COLORS['primary'],
                width=4
            ),
            annotation_text=f"<b>Threshold: {threshold:.3f}</b>",
            annotation=dict(
                font=dict(size=13, color=MODERN_COLORS['dark'], family='Inter'),
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor=MODERN_COLORS['primary'],
                borderwidth=2,
                borderpad=6
            ),
            annotation_position="top right"
        )
        
        # Modern layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, family='Inter', color=MODERN_COLORS['dark']),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="<b>Attack Probability</b>",
            yaxis_title="<b>Sample Count</b>",
            barmode='overlay',
            height=480,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_traffic_monitoring_chart(
        timestamps: List[datetime],
        traffic: np.ndarray,
        attacks: np.ndarray
    ) -> go.Figure:
        """Premium unified traffic monitoring chart"""
        fig = go.Figure()
        
        # Traffic line with gradient fill
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=traffic,
                mode='lines',
                name='Network Traffic',
                line=dict(
                    color=MODERN_COLORS['primary'],
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                fill='tozeroy',
                fillcolor=f"rgba(99, 102, 241, 0.2)",
                hovertemplate='<b>Traffic Volume</b><br>' +
                             'Time: %{x|%H:%M:%S}<br>' +
                             'Packets/sec: %{y:.0f}<extra></extra>',
                showlegend=True,
                yaxis='y1'
            )
        )
        
        # Attack markers on the same chart (using secondary y-axis for visibility)
        attack_times = [timestamps[i] for i in range(len(attacks)) if attacks[i] == 1]
        attack_traffic = [traffic[i] for i in range(len(attacks)) if attacks[i] == 1]
        
        if len(attack_times) > 0:
            fig.add_trace(
                go.Scatter(
                    x=attack_times,
                    y=attack_traffic,
                    mode='markers',
                    name='Attack Detected',
                    marker=dict(
                        color=MODERN_COLORS['danger'],
                        size=16,
                        symbol='x',
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    hovertemplate='<b>🚨 ATTACK DETECTED</b><br>' +
                                 'Time: %{x|%H:%M:%S}<br>' +
                                 'Traffic: %{y:.0f} pkt/s<br>' +
                                 'Severity: High<extra></extra>',
                    showlegend=True,
                    yaxis='y1'
                )
            )
        
        # Threshold line (using rgba for opacity)
        fig.add_hline(
            y=TRAFFIC_THRESHOLD,
            line=dict(
                dash="dash",
                color='rgba(245, 158, 11, 0.6)',  # warning color with opacity
                width=2
            ),
            annotation_text=f"Threshold: {TRAFFIC_THRESHOLD}",
            annotation=dict(
                font=dict(size=11, color=MODERN_COLORS['warning'], family='Inter'),
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor=MODERN_COLORS['warning'],
                borderwidth=1,
                borderpad=4
            ),
            annotation_position="right"
        )
        
        # Modern layout with dark mode support
        fig.update_layout(
            title=dict(
                text="<b>Real-Time Network Traffic & Attack Detection</b>",
                font=dict(size=20, family='Inter', color='white'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text="<b>Time</b>",
                    font=dict(size=13, family='Inter', color='white')
                ),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(148, 163, 184, 0.15)',
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title=dict(
                    text="<b>Traffic (packets/sec)</b>",
                    font=dict(size=13, family='Inter', color='white')
                ),
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(148, 163, 184, 0.15)',
                tickfont=dict(color='white')
            ),
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0, 0, 0, 0.5)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1,
                font=dict(color='white', size=12)
            ),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='Inter', color='white'),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        return fig
    
    @staticmethod
    def create_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        model_name: str,
        title: str = "ROC Curve"
    ) -> go.Figure:
        """Premium ROC curve with modern styling"""
        fig = go.Figure()
        
        # ROC curve with gradient fill
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {auc_score:.3f})',
            line=dict(
                color=MODERN_COLORS['primary'],
                width=4,
                shape='spline'
            ),
            fill='tozeroy',
            fillcolor=f"rgba(99, 102, 241, 0.2)",
            hovertemplate='<b>ROC Curve</b><br>' +
                         'FPR: %{x:.3f}<br>' +
                         'TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(
                color=MODERN_COLORS['gray_light'],
                width=2.5,
                dash='dash'
            ),
            hovertemplate='Random Classifier<extra></extra>'
        ))
        
        # Premium layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title} - {model_name}</b>",
                font=dict(size=22, family='Inter', color=MODERN_COLORS['dark']),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="<b>False Positive Rate</b>",
            yaxis_title="<b>True Positive Rate</b>",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=500,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_precision_recall_curve(
        precision: np.ndarray,
        recall: np.ndarray,
        model_name: str,
        title: str = "Precision-Recall Curve"
    ) -> go.Figure:
        """Premium PR curve"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{model_name}',
            line=dict(
                color=MODERN_COLORS['secondary'],
                width=4,
                shape='spline'
            ),
            fill='tozeroy',
            fillcolor=f"rgba(139, 92, 246, 0.2)",
            hovertemplate='<b>PR Curve</b><br>' +
                         'Recall: %{x:.3f}<br>' +
                         'Precision: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title} - {model_name}</b>",
                font=dict(size=22, family='Inter', color=MODERN_COLORS['dark']),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="<b>Recall</b>",
            yaxis_title="<b>Precision</b>",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=500,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_heatmap(
        cm: np.ndarray,
        labels: List[str] = None,
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """Premium confusion matrix with modern color scheme"""
        if labels is None:
            labels = ['Normal', 'Attack']
        
        # Modern heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont=dict(size=18, color='white', family='Inter'),
            colorscale='Viridis',
            hovertemplate='<b>Actual: %{y}</b><br>' +
                         '<b>Predicted: %{x}</b><br>' +
                         'Count: <b>%{z}</b><extra></extra>',
            colorbar=dict(
                title="<b>Count</b>",
                len=0.8,
                thickness=20,
                titlefont={'size': 14, 'family': 'Inter'},
                tickfont={'size': 12}
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, family='Inter', color=MODERN_COLORS['dark']),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="<b>Predicted Label</b>",
            yaxis_title="<b>True Label</b>",
            height=500,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(
        importance_df: pd.DataFrame,
        top_n: int,
        selected_model: str,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """Premium feature importance with gradient bars"""
        fig = go.Figure()
        
        # Top features
        top_features = importance_df.head(top_n)[::-1]
        
        fig.add_trace(go.Bar(
            y=top_features['Feature'],
            x=top_features['Importance'],
            orientation='h',
            marker=dict(
                color=top_features['Importance'],
                colorscale='Plasma',
                line=dict(color='white', width=2),
                showscale=True,
                colorbar=dict(
                    title="<b>Score</b>",
                    titlefont={'size': 12, 'family': 'Inter'},
                    tickfont={'size': 11}
                )
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         'Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>Top {top_n} Features - {selected_model}</b>",
                font=dict(size=22, family='Inter', color=MODERN_COLORS['dark']),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="<b>Importance Score</b>",
            yaxis_title="<b>Feature</b>",
            height=650,
            showlegend=False,
            **MODERN_LAYOUT
        )
        
        return fig
    
    @staticmethod
    def create_feature_distribution_pie_chart(
        group_counts: pd.Series,
        title: str = "Feature Importance Distribution"
    ) -> go.Figure:
        """Premium donut chart with modern colors"""
        colors = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6']
        
        fig = go.Figure(data=[go.Pie(
            labels=group_counts.index,
            values=group_counts.values,
            hole=0.5,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=3)
            ),
            textinfo='percent+label',
            textfont=dict(size=14, family='Inter', color='white'),
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>' +
                         'Count: %{value}<br>' +
                         'Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, family='Inter', color=MODERN_COLORS['dark']),
                x=0.5,
                xanchor='center'
            ),
            height=500,
            **MODERN_LAYOUT
        )
        
        return fig
