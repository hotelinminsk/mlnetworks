"""
Professional Real-Time Intrusion Detection System Dashboard
SOC-Level Network Security Monitoring
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.realtime_ids_service import RealtimeIDSService

# Page config
st.set_page_config(
    page_title="SOC | Real-Time Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# DARK THEME CSS - FULL PROFESSIONAL
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }

    /* Remove white backgrounds */
    .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }

    /* Headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff 0%, #ff00ea 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
    }

    .sub-header {
        text-align: center;
        color: #8b92b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Metric cards - DARK */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(26,31,58,0.9) 0%, rgba(42,47,68,0.9) 100%);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 1.5rem 1rem;
        box-shadow: 0 8px 32px rgba(0,212,255,0.1);
    }

    div[data-testid="metric-container"] label {
        color: #8b92b0 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #00ff88 !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }

    /* Alert boxes */
    .alert-critical {
        background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.3) 100%);
        border-left: 4px solid #dc2626;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #fca5a5;
        font-weight: 500;
    }

    .alert-high {
        background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.3) 100%);
        border-left: 4px solid #d97706;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #fcd34d;
        font-weight: 500;
    }

    .alert-medium {
        background: linear-gradient(135deg, rgba(251,191,36,0.2) 0%, rgba(245,158,11,0.3) 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #fde68a;
        font-weight: 500;
    }

    /* Traffic feed */
    .traffic-normal {
        background: rgba(16,185,129,0.1);
        border-left: 3px solid #10b981;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        color: #6ee7b7;
        font-size: 0.9rem;
    }

    .traffic-attack {
        background: rgba(239,68,68,0.15);
        border-left: 3px solid #ef4444;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        color: #fca5a5;
        font-size: 0.9rem;
        animation: pulse-red 2s infinite;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 5px rgba(239,68,68,0.3); }
        50% { box-shadow: 0 0 20px rgba(239,68,68,0.6); }
    }

    /* Status indicators */
    .status-active {
        color: #10b981;
        font-weight: 700;
        font-size: 1.3rem;
        text-shadow: 0 0 10px rgba(16,185,129,0.5);
    }

    .status-inactive {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.3rem;
    }

    /* Section headers */
    .section-header {
        color: #00d4ff;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,212,255,0.3);
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: rgba(26,31,58,0.5);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize service
if 'ids_service' not in st.session_state:
    st.session_state.ids_service = None
    st.session_state.is_running = False
    st.session_state.packet_history = deque(maxlen=200)
    st.session_state.alert_history = deque(maxlen=100)
    st.session_state.attack_timeline = deque(maxlen=50)
    st.session_state.traffic_rate = deque(maxlen=30)
    st.session_state.attack_rate_timeline = deque(maxlen=30)


def render_header():
    """Render main header"""
    st.markdown('<div class="main-header">🛡️ NETWORK SECURITY OPERATIONS CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Intrusion Detection & Threat Intelligence Platform</div>', unsafe_allow_html=True)


def render_control_panel():
    """Control panel"""
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2])

    with col1:
        if st.button("🚀 START MONITORING", use_container_width=True, disabled=st.session_state.is_running):
            if st.session_state.ids_service is None:
                st.session_state.ids_service = RealtimeIDSService()
            st.session_state.ids_service.start()
            st.session_state.is_running = True
            st.rerun()

    with col2:
        if st.button("⏸️ STOP MONITORING", use_container_width=True, disabled=not st.session_state.is_running):
            if st.session_state.ids_service:
                st.session_state.ids_service.stop()
            st.session_state.is_running = False
            st.rerun()

    with col3:
        if st.button("🔄 RESET DATA", use_container_width=True):
            if st.session_state.ids_service:
                st.session_state.ids_service.reset()
            st.session_state.packet_history.clear()
            st.session_state.alert_history.clear()
            st.session_state.attack_timeline.clear()
            st.session_state.traffic_rate.clear()
            st.session_state.attack_rate_timeline.clear()
            st.rerun()

    with col4:
        st.selectbox("ML Model", ["gradient_boosting", "xgboost", "lightgbm", "random_forest"], key="model_select", label_visibility="collapsed")

    with col5:
        status_text = "🟢 ACTIVE" if st.session_state.is_running else "🔴 INACTIVE"
        status_class = "status-active" if st.session_state.is_running else "status-inactive"
        st.markdown(f'<div class="{status_class}" style="text-align: center; padding: 0.5rem;">{status_text}</div>', unsafe_allow_html=True)


def render_kpi_dashboard():
    """Main KPI metrics"""
    if not st.session_state.ids_service:
        st.info("🚀 Start monitoring to see real-time statistics")
        return

    stats = st.session_state.ids_service.get_statistics()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "📡 PACKETS",
            f"{stats['packets_processed']:,}",
            f"+{stats['replay_stats'].get('progress_percentage', 0):.0f}%"
        )

    with col2:
        st.metric(
            "🚨 THREATS",
            f"{stats['attacks_detected']:,}",
            f"{stats['alert_summary']['critical']} Critical"
        )

    with col3:
        st.metric(
            "✅ BENIGN",
            f"{stats['normal_traffic']:,}",
            "Verified"
        )

    with col4:
        accuracy = stats.get('accuracy', 0)
        st.metric(
            "🎯 ACCURACY",
            f"{accuracy:.1f}%",
            f"{stats['true_positives']} TP"
        )

    with col5:
        attack_rate = (stats['attacks_detected'] / stats['packets_processed'] * 100) if stats['packets_processed'] > 0 else 0
        st.metric(
            "⚡ THREAT RATE",
            f"{attack_rate:.1f}%",
            "Live"
        )

    with col6:
        st.metric(
            "🔥 CRITICAL",
            f"{stats['alert_summary']['critical']}",
            f"+{stats['alert_summary']['high']} High"
        )


def render_live_charts():
    """Live traffic and attack charts"""
    st.markdown('<div class="section-header">📊 REAL-TIME ANALYTICS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Attack rate over time
        fig = go.Figure()

        if len(st.session_state.attack_rate_timeline) > 0:
            times = list(range(len(st.session_state.attack_rate_timeline)))
            rates = list(st.session_state.attack_rate_timeline)

            fig.add_trace(go.Scatter(
                x=times,
                y=rates,
                mode='lines+markers',
                name='Attack Rate',
                line=dict(color='#ef4444', width=3),
                fill='tozeroy',
                fillcolor='rgba(239,68,68,0.2)'
            ))

        fig.update_layout(
            title="Attack Rate Timeline",
            xaxis_title="Time (samples)",
            yaxis_title="Attack Rate (%)",
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(26,31,58,0.5)',
            plot_bgcolor='rgba(26,31,58,0.8)',
            font=dict(color='#8b92b0')
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Traffic volume
        fig = go.Figure()

        if len(st.session_state.traffic_rate) > 0:
            times = list(range(len(st.session_state.traffic_rate)))
            packets = list(st.session_state.traffic_rate)

            fig.add_trace(go.Bar(
                x=times,
                y=packets,
                name='Packets/sec',
                marker=dict(color='#00d4ff')
            ))

        fig.update_layout(
            title="Traffic Volume",
            xaxis_title="Time (samples)",
            yaxis_title="Packets",
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(26,31,58,0.5)',
            plot_bgcolor='rgba(26,31,58,0.8)',
            font=dict(color='#8b92b0')
        )

        st.plotly_chart(fig, use_container_width=True)


def render_attack_analysis():
    """Attack type and distribution analysis"""
    st.markdown('<div class="section-header">🎯 THREAT INTELLIGENCE</div>', unsafe_allow_html=True)

    if not st.session_state.ids_service:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        # Attack type distribution
        attack_dist = st.session_state.ids_service.get_attack_distribution()

        if not attack_dist.empty:
            fig = px.pie(
                values=attack_dist['count'],
                names=attack_dist.index,
                title="Attack Type Distribution",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                paper_bgcolor='rgba(26,31,58,0.5)',
                font=dict(color='#8b92b0')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attacks detected yet")

    with col2:
        # Severity breakdown
        alert_summary = st.session_state.ids_service.get_statistics()['alert_summary']

        severity_df = pd.DataFrame({
            'Severity': ['Critical', 'High', 'Medium', 'Low'],
            'Count': [
                alert_summary['critical'],
                alert_summary['high'],
                alert_summary['medium'],
                alert_summary['low']
            ]
        })

        fig = go.Figure(data=[
            go.Bar(
                x=severity_df['Severity'],
                y=severity_df['Count'],
                marker=dict(
                    color=['#dc2626', '#f59e0b', '#fbbf24', '#10b981'],
                    line=dict(color='rgba(0,212,255,0.3)', width=1)
                )
            )
        ])

        fig.update_layout(
            title="Alert Severity Distribution",
            template="plotly_dark",
            height=350,
            paper_bgcolor='rgba(26,31,58,0.5)',
            plot_bgcolor='rgba(26,31,58,0.8)',
            font=dict(color='#8b92b0')
        )

        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Top attack sources
        if len(st.session_state.packet_history) > 0:
            attack_packets = [p for p in st.session_state.packet_history if p['prediction'] == 'attack']

            if attack_packets:
                src_ips = [p['srcip'] for p in attack_packets]
                top_sources = pd.Series(src_ips).value_counts().head(8)

                fig = go.Figure(data=[
                    go.Bar(
                        y=top_sources.index,
                        x=top_sources.values,
                        orientation='h',
                        marker=dict(color='#ef4444')
                    )
                ])

                fig.update_layout(
                    title="Top Attack Sources",
                    template="plotly_dark",
                    height=350,
                    paper_bgcolor='rgba(26,31,58,0.5)',
                    plot_bgcolor='rgba(26,31,58,0.8)',
                    font=dict(color='#8b92b0')
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No attack sources yet")
        else:
            st.info("Waiting for data...")


def render_traffic_feed():
    """Live traffic feed"""
    st.markdown('<div class="section-header">🔴 LIVE TRAFFIC STREAM</div>', unsafe_allow_html=True)

    if not st.session_state.is_running:
        st.warning("⚠️ Start monitoring to see live traffic")
        return

    # Process packets
    for _ in range(10):
        packet = st.session_state.ids_service.process_next_packet()
        if packet:
            st.session_state.packet_history.append(packet)
            if packet['alert']:
                st.session_state.alert_history.append(packet['alert'])

            # Update timelines
            if len(st.session_state.packet_history) >= 30:
                recent_30 = list(st.session_state.packet_history)[-30:]
                attack_count = sum(1 for p in recent_30 if p['prediction'] == 'attack')
                attack_rate = (attack_count / 30) * 100
                st.session_state.attack_rate_timeline.append(attack_rate)
                st.session_state.traffic_rate.append(len(recent_30))

    # Show recent packets
    recent = list(st.session_state.packet_history)[-15:]

    for pkt in reversed(recent):
        is_attack = pkt['prediction'] == 'attack'
        css_class = "traffic-attack" if is_attack else "traffic-normal"
        icon = "🚨" if is_attack else "✅"

        attack_info = f" | <strong style='color: #ff00ea;'>{pkt['attack_type']}</strong>" if pkt.get('attack_type') and pkt['attack_type'] != 'Normal' else ""

        st.markdown(f"""
        <div class="{css_class}">
            {icon} <strong>{pkt['timestamp'].strftime('%H:%M:%S.%f')[:-3]}</strong> |
            <span style='color: #00d4ff;'>{pkt['srcip']}</span> →
            <span style='color: #00ff88;'>{pkt['dstip']}</span> |
            Service: <strong>{pkt['service']}</strong> |
            Confidence: <strong>{pkt['probability']:.1%}</strong>
            {attack_info}
        </div>
        """, unsafe_allow_html=True)


def render_alerts():
    """Security alerts panel"""
    st.markdown('<div class="section-header">🚨 SECURITY ALERTS</div>', unsafe_allow_html=True)

    if not st.session_state.ids_service:
        return

    alerts = st.session_state.ids_service.get_recent_alerts(12)

    if not alerts:
        st.success("✅ No security threats detected - System secure")
        return

    for alert in reversed(alerts):
        severity = alert['severity']
        alert_class = f"alert-{severity}"

        icon_map = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        icon = icon_map.get(severity, '⚪')

        st.markdown(f"""
        <div class="{alert_class}">
            {icon} <strong style='font-size: 1.1rem;'>[{severity.upper()}]</strong> {alert['message']}<br>
            <small style='opacity: 0.8;'>
                🎯 Confidence: {alert['probability']:.1%} |
                🕐 {alert['timestamp'].strftime('%H:%M:%S')} |
                🔢 Alert #{alert['id']}
            </small>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    render_header()
    render_control_panel()

    st.markdown("---")

    render_kpi_dashboard()

    st.markdown("---")

    render_live_charts()

    st.markdown("---")

    render_attack_analysis()

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        render_traffic_feed()

    with col2:
        render_alerts()

    # Auto-refresh
    if st.session_state.is_running:
        time.sleep(0.3)
        st.rerun()


if __name__ == "__main__":
    main()
