"""
ULTIMATE NETWORK IDS DASHBOARD
Complete presentation-ready system with ALL features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
from pathlib import Path
import sys
from collections import deque
from sklearn.metrics import confusion_matrix, roc_curve, auc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.realtime_ids_service import RealtimeIDSService
from services.multi_model_service import MultiModelService
from src.config import DATA_PROCESSED, MODELS

# Page config
st.set_page_config(
    page_title="Ultimate Network IDS | Full Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PROFESSIONAL DARK THEME
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    .block-container { background: transparent !important; padding-top: 1rem !important; }

    .mega-header {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff 0%, #ff00ea 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 30px rgba(0,212,255,0.6);
    }

    .sub-title {
        text-align: center;
        color: #8b92b0;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26,31,58,0.5);
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(102,126,234,0.2);
        border-radius: 8px;
        color: #8b92b0;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(26,31,58,0.9) 0%, rgba(42,47,68,0.9) 100%);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 1.5rem 1rem;
        box-shadow: 0 8px 32px rgba(0,212,255,0.1);
    }

    div[data-testid="metric-container"] label {
        color: #8b92b0 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 1.8rem !important;
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
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    /* Section headers */
    .section-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,212,255,0.3);
    }

    /* Traffic feed */
    .traffic-attack {
        background: rgba(239,68,68,0.15);
        border-left: 3px solid #ef4444;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        color: #fca5a5;
        animation: pulse-red 2s infinite;
    }

    .traffic-normal {
        background: rgba(16,185,129,0.1);
        border-left: 3px solid #10b981;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        color: #6ee7b7;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 5px rgba(239,68,68,0.3); }
        50% { box-shadow: 0 0 20px rgba(239,68,68,0.6); }
    }

    /* Hide streamlit branding */
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

# Initialize services
if 'ids_service' not in st.session_state:
    st.session_state.ids_service = None
    st.session_state.multi_model = MultiModelService()
    st.session_state.is_running = False
    st.session_state.packet_history = deque(maxlen=500)
    st.session_state.alert_history = deque(maxlen=200)
    st.session_state.model_predictions = deque(maxlen=100)
    st.session_state.attack_timeline = deque(maxlen=50)
    st.session_state.protocol_stats = {}
    st.session_state.port_stats = {}

# Load test data for metrics
@st.cache_data
def load_test_data():
    X = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values
    return X, y


def render_header():
    """Main header"""
    st.markdown('<div class="mega-header">🛡️ ULTIMATE NETWORK IDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Complete Intrusion Detection Suite | Real-Time Monitoring & ML Intelligence</div>', unsafe_allow_html=True)


def render_control_panel():
    """Control buttons"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 START SYSTEM", use_container_width=True, disabled=st.session_state.is_running):
            if st.session_state.ids_service is None:
                st.session_state.ids_service = RealtimeIDSService()
            st.session_state.ids_service.start()
            st.session_state.is_running = True
            st.rerun()

    with col2:
        if st.button("⏸️ STOP SYSTEM", use_container_width=True, disabled=not st.session_state.is_running):
            if st.session_state.ids_service:
                st.session_state.ids_service.stop()
            st.session_state.is_running = False
            st.rerun()

    with col3:
        if st.button("🔄 RESET ALL", use_container_width=True):
            if st.session_state.ids_service:
                st.session_state.ids_service.reset()
            st.session_state.packet_history.clear()
            st.session_state.alert_history.clear()
            st.session_state.model_predictions.clear()
            st.rerun()

    with col4:
        status = "🟢 ACTIVE" if st.session_state.is_running else "🔴 INACTIVE"
        st.markdown(f"<div style='text-align: center; padding: 0.5rem; font-size: 1.3rem; font-weight: 700; color: {'#10b981' if st.session_state.is_running else '#ef4444'};'>{status}</div>", unsafe_allow_html=True)


def tab_realtime_monitoring():
    """Tab 1: Real-Time Monitoring"""
    st.markdown('<div class="section-title">📡 REAL-TIME NETWORK MONITORING</div>', unsafe_allow_html=True)

    if not st.session_state.ids_service or not st.session_state.is_running:
        st.info("🚀 Click START SYSTEM to begin real-time monitoring")
        return

    # KPIs
    stats = st.session_state.ids_service.get_statistics()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 PACKETS", f"{stats['packets_processed']:,}", f"+{len(st.session_state.packet_history)}")
    with col2:
        st.metric("🚨 THREATS", f"{stats['attacks_detected']:,}", f"{stats['alert_summary']['critical']} Critical")
    with col3:
        st.metric("✅ BENIGN", f"{stats['normal_traffic']:,}", "Verified")
    with col4:
        accuracy = stats.get('accuracy', 0)
        st.metric("🎯 ACCURACY", f"{accuracy:.1f}%", f"{stats['true_positives']} TP")
    with col5:
        attack_rate = (stats['attacks_detected'] / stats['packets_processed'] * 100) if stats['packets_processed'] > 0 else 0
        st.metric("⚡ THREAT RATE", f"{attack_rate:.1f}%", "Live")

    st.markdown("---")

    # Process packets
    for _ in range(10):
        packet = st.session_state.ids_service.process_next_packet()
        if packet:
            st.session_state.packet_history.append(packet)

            # Protocol stats
            proto = packet.get('proto', 'Unknown')
            st.session_state.protocol_stats[proto] = st.session_state.protocol_stats.get(proto, 0) + 1

            # Port stats
            service = packet.get('service', 'Unknown')
            st.session_state.port_stats[service] = st.session_state.port_stats.get(service, 0) + 1

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Traffic feed
        st.markdown("### 🔴 LIVE TRAFFIC STREAM")
        recent = list(st.session_state.packet_history)[-12:]
        for pkt in reversed(recent):
            is_attack = pkt['prediction'] == 'attack'
            css_class = "traffic-attack" if is_attack else "traffic-normal"
            icon = "🚨" if is_attack else "✅"

            attack_info = f" | <strong>{pkt['attack_type']}</strong>" if pkt.get('attack_type') and pkt['attack_type'] != 'Normal' else ""

            st.markdown(f"""
            <div class="{css_class}">
                {icon} {pkt['timestamp'].strftime('%H:%M:%S')} |
                {pkt['srcip']} → {pkt['dstip']} |
                {pkt['service']} | {pkt['probability']:.1%}
                {attack_info}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Alert feed
        st.markdown("### 🚨 SECURITY ALERTS")
        alerts = st.session_state.ids_service.get_recent_alerts(10)
        if alerts:
            for alert in reversed(alerts):
                severity_color = {'critical': '#dc2626', 'high': '#f59e0b', 'medium': '#fbbf24', 'low': '#10b981'}
                color = severity_color.get(alert['severity'], '#8b92b0')

                st.markdown(f"""
                <div style='background: {color}22; border-left: 4px solid {color}; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; color: {color};'>
                    [{alert['severity'].upper()}] {alert['message']}<br>
                    <small>Confidence: {alert['probability']:.1%} | {alert['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No threats detected")


def tab_model_comparison():
    """Tab 2: Multi-Model Comparison"""
    st.markdown('<div class="section-title">🤖 MULTI-MODEL COMPARISON</div>', unsafe_allow_html=True)

    st.info(f"📊 **{len(st.session_state.multi_model.get_available_models())} Models Loaded**: " +
            ", ".join([m.replace('_', ' ').title() for m in st.session_state.multi_model.get_available_models()]))

    # Test sample selection
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_idx = st.number_input("Select Test Sample Index", min_value=0, max_value=10000, value=100, step=1)
    with col2:
        if st.button("🔍 ANALYZE SAMPLE", use_container_width=True):
            X_test, y_test = load_test_data()
            sample = X_test.iloc[sample_idx:sample_idx+1]
            true_label = y_test[sample_idx]

            predictions = st.session_state.multi_model.predict_all(sample)
            agreement = st.session_state.multi_model.get_model_agreement(predictions)

            st.session_state.current_predictions = predictions
            st.session_state.current_agreement = agreement
            st.session_state.current_true_label = true_label

    if hasattr(st.session_state, 'current_predictions'):
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 TRUE LABEL", "ATTACK" if st.session_state.current_true_label == 1 else "NORMAL")

        with col2:
            st.metric("🤝 CONSENSUS", st.session_state.current_agreement['consensus'].upper(),
                     f"{st.session_state.current_agreement['agreement_percentage']:.0f}% agreement")

        with col3:
            st.metric("📊 VOTES",
                     f"Attack: {st.session_state.current_agreement['attack_votes']} | Normal: {st.session_state.current_agreement['normal_votes']}")

        st.markdown("---")

        # Model predictions table
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 MODEL PREDICTIONS")
            comparison_df = []
            for model_name, result in st.session_state.current_predictions.items():
                comparison_df.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Prediction': result['prediction'].upper(),
                    'Probability': f"{result['probability']:.2%}",
                    'Confidence': result['confidence'].upper()
                })

            st.dataframe(pd.DataFrame(comparison_df), use_container_width=True, height=350)

        with col2:
            st.markdown("### 📊 PROBABILITY DISTRIBUTION")
            probs = [(m.replace('_', ' ').title(), r['probability']) for m, r in st.session_state.current_predictions.items()]
            prob_df = pd.DataFrame(probs, columns=['Model', 'Probability'])

            fig = px.bar(prob_df, x='Model', y='Probability',
                        color='Probability',
                        color_continuous_scale='RdYlGn_r',
                        title="Model Confidence Comparison")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(26,31,58,0.5)',
                plot_bgcolor='rgba(26,31,58,0.8)',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


def tab_anomaly_detection():
    """Tab 3: Anomaly Detection (Isolation Forest Focus)"""
    st.markdown('<div class="section-title">🔬 ANOMALY DETECTION ANALYSIS</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Isolation Forest - Unsupervised Anomaly Detection

    **Key Features:**
    - Detects anomalies without labeled data
    - Based on decision tree isolation
    - Lower anomaly score = higher anomaly likelihood
    - Effective for unknown attack types
    """)

    # Load sample data for visualization
    X_test, y_test = load_test_data()
    sample_size = st.slider("Sample Size for Analysis", 100, 1000, 500, 50)

    if st.button("🔍 RUN ANOMALY ANALYSIS"):
        with st.spinner("Analyzing anomalies..."):
            samples = X_test.iloc[:sample_size]
            labels = y_test[:sample_size]

            # Get isolation forest predictions
            if 'isolation_forest' in st.session_state.multi_model.models:
                iforest = st.session_state.multi_model.models['isolation_forest']

                # Get anomaly scores
                X_prep = st.session_state.multi_model.preprocessor.transform(samples)
                scores = iforest.score_samples(X_prep)
                predictions = iforest.predict(X_prep)

                # -1 = anomaly, 1 = normal in isolation forest
                predictions_binary = (predictions == -1).astype(int)

                st.session_state.anomaly_scores = scores
                st.session_state.anomaly_preds = predictions_binary
                st.session_state.anomaly_labels = labels

    if hasattr(st.session_state, 'anomaly_scores'):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Anomaly Score Distribution")

            fig = go.Figure()

            # Normal samples
            normal_scores = st.session_state.anomaly_scores[st.session_state.anomaly_labels == 0]
            fig.add_trace(go.Histogram(x=normal_scores, name='Normal', marker_color='#10b981', opacity=0.7))

            # Attack samples
            attack_scores = st.session_state.anomaly_scores[st.session_state.anomaly_labels == 1]
            fig.add_trace(go.Histogram(x=attack_scores, name='Attacks', marker_color='#ef4444', opacity=0.7))

            fig.update_layout(
                title="Anomaly Score Distribution",
                xaxis_title="Anomaly Score",
                yaxis_title="Count",
                template="plotly_dark",
                barmode='overlay',
                paper_bgcolor='rgba(26,31,58,0.5)',
                plot_bgcolor='rgba(26,31,58,0.8)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Detection Performance")

            # Confusion matrix
            cm = confusion_matrix(st.session_state.anomaly_labels, st.session_state.anomaly_preds)

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Attack'],
                y=['Normal', 'Attack'],
                colorscale='RdYlGn_r',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))

            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                template="plotly_dark",
                paper_bgcolor='rgba(26,31,58,0.5)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("🎪 Precision", f"{precision:.2%}")
        with col3:
            st.metric("🔍 Recall", f"{recall:.2%}")
        with col4:
            st.metric("⚖️ F1 Score", f"{f1:.2%}")


def tab_network_analysis():
    """Tab 4: Network Traffic Analysis"""
    st.markdown('<div class="section-title">🌐 NETWORK TRAFFIC ANALYSIS</div>', unsafe_allow_html=True)

    if len(st.session_state.packet_history) == 0:
        st.info("Start system to collect network data")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📡 Protocol Distribution")
        if st.session_state.protocol_stats:
            proto_df = pd.DataFrame(list(st.session_state.protocol_stats.items()), columns=['Protocol', 'Count'])
            fig = px.pie(proto_df, values='Count', names='Protocol',
                        color_discrete_sequence=px.colors.sequential.Plasma)
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🔌 Top Services")
        if st.session_state.port_stats:
            port_df = pd.DataFrame(list(st.session_state.port_stats.items()), columns=['Service', 'Count'])
            port_df = port_df.nlargest(10, 'Count')
            fig = px.bar(port_df, x='Service', y='Count', color='Count',
                        color_continuous_scale='Viridis')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)',
                            plot_bgcolor='rgba(26,31,58,0.8)', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### 🎯 Attack Types")
        attack_packets = [p for p in st.session_state.packet_history if p['prediction'] == 'attack' and p.get('attack_type')]
        if attack_packets:
            attack_types = [p['attack_type'] for p in attack_packets]
            attack_dist = pd.Series(attack_types).value_counts()
            fig = px.bar(x=attack_dist.index, y=attack_dist.values,
                        color=attack_dist.values,
                        color_continuous_scale='Reds')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)',
                            plot_bgcolor='rgba(26,31,58,0.8)', height=300)
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application"""
    render_header()
    render_control_panel()

    st.markdown("---")

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 REAL-TIME MONITORING",
        "🤖 MODEL COMPARISON",
        "🔬 ANOMALY DETECTION",
        "🌐 NETWORK ANALYSIS"
    ])

    with tab1:
        tab_realtime_monitoring()

    with tab2:
        tab_model_comparison()

    with tab3:
        tab_anomaly_detection()

    with tab4:
        tab_network_analysis()

    # Auto-refresh
    if st.session_state.is_running:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
