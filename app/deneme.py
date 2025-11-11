"""
FINAL COMPLETE NETWORK IDS DASHBOARD
Presentation-Ready System with ALL Features + Attack Intelligence
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
from textwrap import dedent
from sklearn.metrics import confusion_matrix, roc_curve, auc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.realtime_ids_service import RealtimeIDSService
from services.multi_model_service import MultiModelService
from src.config import DATA_PROCESSED, MODELS

# Page config
st.set_page_config(
    page_title="Network IDS | Complete Suite",
    page_icon="IDS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PROFESSIONAL DARK THEME + ATTACK DETAIL STYLING
st.markdown("""

<style>
    .stApp {
        background: linear-gradient(140deg, #0f172a 0%, #1f2937 70%);
        color: #e5e7eb;
    }

    .main .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
        background: transparent !important;
    }

    .mega-header {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        color: #e5e7eb;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.65);
        padding: 0.4rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(59, 130, 246, 0.15);
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.7rem 1.3rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #2563eb 0%, #1d4ed8 100%);
        color: #f8fafc;
    }

    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
    }

    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e5e7eb !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    .stButton button {
        background: linear-gradient(120deg, #2563eb 0%, #1d4ed8 100%);
        color: #f8fafc;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 12px 20px rgba(37, 99, 235, 0.25);
    }

    .section-title {
        color: #d1d5db;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.3);
    }

    .attack-detail {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
    }

    .attack-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fca5a5;
        margin-bottom: 1rem;
    }

    .attack-info-row {
        background: rgba(15, 23, 42, 0.65);
        padding: 0.9rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid rgba(248, 113, 113, 0.6);
    }

    .attack-label {
        color: #fda4af;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .attack-value {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    .traffic-feed {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .traffic-row {
        display: grid;
        grid-template-columns: 110px 1fr 1fr 90px 110px 110px;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        align-items: center;
        font-size: 0.9rem;
    }

    .traffic-row + .traffic-row {
        border-top: 1px solid rgba(148, 163, 184, 0.15);
    }

    .traffic-row.header {
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        color: #94a3b8;
        background: rgba(15, 23, 42, 0.95);
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }

    .traffic-row.attack {
        background: rgba(248, 113, 113, 0.08);
        border-left: 4px solid rgba(248, 113, 113, 0.8);
    }

    .traffic-row.normal {
        background: rgba(34, 197, 94, 0.05);
        border-left: 4px solid rgba(34, 197, 94, 0.7);
    }

    .traffic-cell {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .traffic-status {
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .traffic-status.attack { color: #f87171; }
    .traffic-status.normal { color: #34d399; }

    .alert-feed {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }

    .alert-card {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid #38bdf8;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
    }

    .alert-card.critical { border-color: #ef4444; }
    .alert-card.high { border-color: #f97316; }
    .alert-card.medium { border-color: #fbbf24; }
    .alert-card.low { border-color: #22c55e; }

    .alert-card-title {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .alert-card-meta {
        font-size: 0.75rem;
        color: #cbd5f5;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    ::-webkit-scrollbar {
        width: 10px;
        background: rgba(30, 41, 59, 0.6);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(120deg, #2563eb 0%, #1d4ed8 100%);
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
    st.session_state.latest_attack = None  # ADDED: Track latest attack

# Load test data for metrics
@st.cache_data
def load_test_data():
    X = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values
    return X, y


def _format_timestamp(value, fmt="%H:%M:%S"):
    """Safely format timestamps coming from replayed packets/alerts."""
    if isinstance(value, datetime):
        return value.strftime(fmt)
    if value is None:
        return "N/A"
    try:
        return pd.to_datetime(value).strftime(fmt)
    except Exception:
        return str(value)


def render_html_block(html: str):
    """Helper to render multi-line HTML without Markdown indent issues."""
    cleaned = dedent(html).strip()
    if cleaned:
        st.markdown(cleaned, unsafe_allow_html=True)


def build_packet_row(packet: dict) -> str:
    """Return HTML for a single traffic row."""
    is_attack = packet.get('prediction') == 'attack'
    status_class = "attack" if is_attack else "normal"
    attack_label = packet.get('attack_type')
    status_label = (attack_label or "ATTACK").upper() if is_attack else "NORMAL"

    probability = packet.get('probability')
    probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "N/A"

    return dedent(f"""
    <div class="traffic-row {status_class}">
        <div class="traffic-cell">{_format_timestamp(packet.get('timestamp'))}</div>
        <div class="traffic-cell">{packet.get('srcip', 'N/A')}</div>
        <div class="traffic-cell">{packet.get('dstip', 'N/A')}</div>
        <div class="traffic-cell">{packet.get('service', '-')}</div>
        <div class="traffic-cell traffic-status {status_class}">{status_label}</div>
        <div class="traffic-cell">{probability_text}</div>
    </div>
    """)


def build_alert_card(alert: dict) -> str:
    """Return HTML for a single alert card."""
    severity = alert.get('severity', 'low')
    probability = alert.get('probability')
    probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "N/A"

    return dedent(f"""
    <div class="alert-card {severity}">
        <div class="alert-card-title">[{severity.upper()}] {alert.get('attack_type', 'Unknown')}</div>
        <div>{alert.get('message', 'Alert generated')}</div>
        <div class="alert-card-meta">
            Confidence: {probability_text} | {_format_timestamp(alert.get('timestamp'))}
        </div>
    </div>
    """)


# ADDED FROM PRO_FIXED: Attack explanation system
def get_attack_explanation(attack_type: str) -> dict:
    """Get detailed explanation for attack type"""
    explanations = {
        'Backdoor': {
            'what': 'Backdoor Attack - Unauthorized remote access attempt',
            'why_dangerous': 'Attacker trying to install persistent remote access mechanism',
            'indicators': 'Unusual port usage, suspicious connection patterns, command execution attempts',
            'action': 'IMMEDIATE: Block source IP, scan for malware, check system for backdoors'
        },
        'Exploits': {
            'what': 'Exploitation Attack - Attempting to exploit system vulnerabilities',
            'why_dangerous': 'Trying to gain unauthorized access by exploiting software flaws',
            'indicators': 'Buffer overflow attempts, malformed packets, privilege escalation patterns',
            'action': 'Block traffic, patch vulnerable systems, review security logs'
        },
        'DoS': {
            'what': 'Denial of Service - Attempt to overwhelm system resources',
            'why_dangerous': 'Can make services unavailable, cause system crashes',
            'indicators': 'High packet rate, repeated connection attempts, resource exhaustion',
            'action': 'Enable rate limiting, block attacking IPs, activate DDoS protection'
        },
        'Reconnaissance': {
            'what': 'Reconnaissance/Scanning - Attacker gathering system information',
            'why_dangerous': 'Preparation phase for future attacks, mapping network vulnerabilities',
            'indicators': 'Port scanning, service enumeration, network mapping attempts',
            'action': 'Monitor closely, log all attempts, consider blocking probing IPs'
        },
        'Shellcode': {
            'what': 'Shellcode Injection - Malicious code execution attempt',
            'why_dangerous': 'Can lead to complete system compromise, data theft',
            'indicators': 'Unusual byte patterns, executable code in data streams',
            'action': 'CRITICAL: Isolate system, scan for malware, investigate payload'
        },
        'Worms': {
            'what': 'Worm Propagation - Self-replicating malware spreading',
            'why_dangerous': 'Can rapidly spread across network, infect multiple systems',
            'indicators': 'Repeated connection attempts to multiple hosts, file replication',
            'action': 'URGENT: Quarantine affected systems, block propagation, run antivirus'
        },
        'Fuzzers': {
            'what': 'Fuzzing Attack - Testing system with malformed inputs',
            'why_dangerous': 'Looking for vulnerabilities to exploit later',
            'indicators': 'Random/malformed data, boundary testing, crash attempts',
            'action': 'Log patterns, validate inputs, update security rules'
        },
        'Analysis': {
            'what': 'Traffic Analysis - Monitoring network for information',
            'why_dangerous': 'Can reveal sensitive information, communication patterns',
            'indicators': 'Passive monitoring, traffic sniffing, pattern analysis',
            'action': 'Enable encryption, monitor suspicious activity'
        },
        'Generic': {
            'what': 'Generic Malicious Activity - Suspicious behavior detected',
            'why_dangerous': 'Unknown threat pattern, potential new attack vector',
            'indicators': 'Anomalous traffic patterns, unusual protocols',
            'action': 'Investigate thoroughly, collect samples for analysis'
        }
    }
    return explanations.get(attack_type, {
        'what': f'{attack_type} Attack',
        'why_dangerous': 'Malicious activity detected',
        'indicators': 'Suspicious network behavior',
        'action': 'Monitor and investigate'
    })


def render_attack_detail(attack_packet: dict):
    """Render detailed attack information - ADDED FROM PRO_FIXED"""
    attack_type = attack_packet.get('attack_type', 'Unknown')
    explanation = get_attack_explanation(attack_type)
    detection_time = attack_packet.get('timestamp')
    detection_str = _format_timestamp(detection_time, "%Y-%m-%d %H:%M:%S.%f")
    if "." in detection_str and detection_str != "N/A":
        detection_str = detection_str[:-3]

    src_ip = attack_packet.get('srcip', 'N/A')
    dst_ip = attack_packet.get('dstip', 'N/A')
    service = attack_packet.get('service', '-')
    probability = attack_packet.get('probability')
    probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "N/A"

    html_lines = [
        '<div class="attack-detail">',
        f'<div class="attack-header">ATTACK DETECTED: {attack_type.upper()}</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">WHEN (Detection Time)</div>',
        f'<div class="attack-value">{detection_str}</div>',
        '</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">WHAT (Attack Type)</div>',
        f'<div class="attack-value">{explanation["what"]}</div>',
        '</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">WHY DANGEROUS</div>',
        f'<div class="attack-value">{explanation["why_dangerous"]}</div>',
        '</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">HOW DETECTED (Indicators)</div>',
        f'<div class="attack-value">{explanation["indicators"]}</div>',
        '</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">SOURCE & TARGET</div>',
        '<div class="attack-value">',
        f'From: <span style="color: #ef4444;">{src_ip}</span> -&gt;',
        f'To: <span style="color: #fbbf24;">{dst_ip}</span> |',
        f'Service: <span style="color: #00d4ff;">{service}</span>',
        '</div>',
        '</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">MODEL CONFIDENCE</div>',
        f'<div class="attack-value">{probability_text} - Model assessment confidence</div>',
        '</div>',
        '<div class="attack-info-row">',
        '<div class="attack-label">RECOMMENDED ACTION</div>',
        f'<div class="attack-value" style="color: #fca5a5; font-weight: 800;">{explanation["action"]}</div>',
        '</div>',
        '</div>',
    ]

    render_html_block("\n".join(html_lines))


def render_header():
    """Main header"""
    st.markdown('<div class="mega-header">Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Threat Detection & Analysis • UNSW-NB15 Dataset</div>', unsafe_allow_html=True)


def render_control_panel():
    """Control buttons"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Start System", use_container_width=True, disabled=st.session_state.is_running):
            if st.session_state.ids_service is None:
                st.session_state.ids_service = RealtimeIDSService()
            st.session_state.ids_service.start()
            st.session_state.is_running = True
            st.rerun()

    with col2:
        if st.button("Stop System", use_container_width=True, disabled=not st.session_state.is_running):
            if st.session_state.ids_service:
                st.session_state.ids_service.stop()
            st.session_state.is_running = False
            st.rerun()

    with col3:
        if st.button("Reset All", use_container_width=True):
            # Stop system if running
            if st.session_state.ids_service:
                st.session_state.ids_service.stop()
                st.session_state.ids_service.reset()

            # Clear all session state
            st.session_state.is_running = False
            st.session_state.packet_history.clear()
            st.session_state.alert_history.clear()
            st.session_state.model_predictions.clear()
            st.session_state.latest_attack = None
            st.session_state.protocol_stats.clear()
            st.session_state.port_stats.clear()

            # Reinitialize services to pick up updated model list
            st.session_state.ids_service = None
            st.session_state.multi_model = MultiModelService()

            # Clear anomaly and comparison results
            if hasattr(st.session_state, 'anomaly_results'):
                delattr(st.session_state, 'anomaly_results')
            if hasattr(st.session_state, 'current_predictions'):
                delattr(st.session_state, 'current_predictions')
            if hasattr(st.session_state, 'current_agreement'):
                delattr(st.session_state, 'current_agreement')

            st.rerun()

    with col4:
        status = "ACTIVE" if st.session_state.is_running else "INACTIVE"
        st.markdown(f"<div style='text-align: center; padding: 0.5rem; font-size: 1.3rem; font-weight: 700; color: {'#10b981' if st.session_state.is_running else '#ef4444'};'>{status}</div>", unsafe_allow_html=True)


def tab_realtime_monitoring():
    """Tab 1: Real-Time Monitoring - ENHANCED WITH ATTACK DETAILS"""
    st.markdown('<div class="section-title">Real-Time Network Monitoring</div>', unsafe_allow_html=True)

    if not st.session_state.ids_service or not st.session_state.is_running:
        st.info("Click Start System to begin real-time monitoring.")

        # Model selection even when not running
        st.markdown("---")
        st.markdown("### System Configuration")

        col1, col2 = st.columns(2)
        with col1:
            available_models = st.session_state.multi_model.get_available_models()
            selected_model = st.selectbox(
                "Select Detection Model",
                options=available_models,
                format_func=lambda x: x.replace('_', ' ').title(),
                index=0 if 'gradient_boosting' in available_models else 0,
                key='model_selector'
            )

            if st.button("Apply Model", use_container_width=True):
                # Re-initialize service with selected model
                st.session_state.ids_service = RealtimeIDSService(model_name=selected_model)
                st.success(f"Model changed to: {selected_model.replace('_', ' ').title()}")

        with col2:
            # Dynamic model list based on what's actually loaded
            model_list = "\n            ".join([f"- {m.replace('_', ' ').title()}" for m in available_models])
            st.info(f"""
            **Current Model:** {selected_model.replace('_', ' ').title()}

            **Available Models ({len(available_models)}):**
            {model_list}
            """)

        return

    # Model selection when running
    with st.expander("Model Selection", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            available_models = st.session_state.multi_model.get_available_models()
            current_model = st.session_state.ids_service.model_name if hasattr(st.session_state.ids_service, 'model_name') else 'gradient_boosting'

            selected_model = st.selectbox(
                "Active Detection Model",
                options=available_models,
                format_func=lambda x: x.replace('_', ' ').title(),
                index=available_models.index(current_model) if current_model in available_models else 0,
                key='model_selector_running'
            )

            if st.button("Change Model (Will Restart)", use_container_width=True):
                st.session_state.ids_service.stop()
                st.session_state.ids_service = RealtimeIDSService(model_name=selected_model)
                st.session_state.ids_service.start()
                st.success(f"Switched to: {selected_model.replace('_', ' ').title()}")
                st.rerun()

        with col2:
            st.info(f"**Current:** {current_model.replace('_', ' ').title()}")

    # KPIs
    stats = st.session_state.ids_service.get_statistics()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Packets", f"{stats['packets_processed']:,}", f"+{len(st.session_state.packet_history)}")
    with col2:
        st.metric("Threats", f"{stats['attacks_detected']:,}", f"{stats['alert_summary']['critical']} Critical")
    with col3:
        st.metric("Benign", f"{stats['normal_traffic']:,}", "Verified")
    with col4:
        accuracy = stats.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.1f}%", f"{stats['true_positives']} TP")
    with col5:
        attack_rate = (stats['attacks_detected'] / stats['packets_processed'] * 100) if stats['packets_processed'] > 0 else 0
        st.metric("Threat Rate", f"{attack_rate:.1f}%", "Live")

    st.markdown("---")

    # Process packets
    for _ in range(10):
        packet = st.session_state.ids_service.process_next_packet()
        if packet:
            st.session_state.packet_history.append(packet)

            # ADDED: Track latest attack for detailed display
            if packet['prediction'] == 'attack' and packet.get('attack_type') and packet['attack_type'] != 'Normal':
                st.session_state.latest_attack = packet

            # Protocol stats
            proto = packet.get('proto', 'Unknown')
            st.session_state.protocol_stats[proto] = st.session_state.protocol_stats.get(proto, 0) + 1

            # Port stats
            service = packet.get('service', 'Unknown')
            st.session_state.port_stats[service] = st.session_state.port_stats.get(service, 0) + 1

    # ADDED: Latest Attack Detail Panel
    if st.session_state.latest_attack:
        st.markdown("### LATEST ATTACK DETECTED - FULL ANALYSIS")
        render_attack_detail(st.session_state.latest_attack)
        st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Traffic feed
        st.markdown("### Live Traffic Stream")
        recent = list(st.session_state.packet_history)[-15:]
        if recent:
            rows_html = "".join(build_packet_row(pkt) for pkt in reversed(recent))
            render_html_block(f"""
            <div class="traffic-feed">
                <div class="traffic-row header">
                    <div class="traffic-cell">Time</div>
                    <div class="traffic-cell">Source IP</div>
                    <div class="traffic-cell">Destination IP</div>
                    <div class="traffic-cell">Service</div>
                    <div class="traffic-cell">Status</div>
                    <div class="traffic-cell">Confidence</div>
                </div>
                {rows_html}
            </div>
            """)
        else:
            st.info("Waiting for packets...")

    with col2:
        # Alert feed
        st.markdown("### Security Alerts")
        alerts = st.session_state.ids_service.get_recent_alerts(10)
        if alerts:
            cards_html = "".join(build_alert_card(alert) for alert in reversed(alerts))
            render_html_block(f"""
            <div class="alert-feed">
                {cards_html}
            </div>
            """)
        else:
            st.success("No threats detected")


def tab_model_comparison():
    """Tab 2: Multi-Model Comparison"""
    num_models = len(st.session_state.multi_model.get_available_models())
    st.markdown(f'<div class="section-title">Multi-Model Comparison ({num_models} Models)</div>', unsafe_allow_html=True)

    # Model descriptions
    model_descriptions = {
        'Gradient Boosting': 'Tree-based ensemble model that builds models sequentially to correct errors',
        'Random Forest': 'Forest of decision trees trained on random data subsets for robust predictions',
        'Extra Trees': 'Extremely randomized trees with faster training and better generalization',
        'Supervised Sgd': 'Fast linear model using stochastic gradient descent',
        'Isolation Forest': 'Unsupervised anomaly detector that isolates outliers quickly'
    }

    model_names = [m.replace('_', ' ').title() for m in st.session_state.multi_model.get_available_models()]

    with st.expander("How to read this comparison", expanded=False):
        st.markdown("""
        **Understanding the metrics**
        - **Prediction**: Model's final decision (ATTACK or NORMAL)
        - **Attack Probability**: Likelihood this is an attack (0% = definitely normal, 100% = definitely attack)
        - **Confidence**: How sure the model is
            - **HIGH**: Probability < 30% or > 70% (clear decision)
            - **MEDIUM**: Probability between 30-70% (uncertain)
            - **LOW**: Probability near 50% (can't decide)

        **Model consensus**
        - If most models agree -> High confidence in final decision
        - If models disagree -> Sample is difficult to classify

        **Chart colors**
        - Green (low %): Model thinks it's NORMAL traffic
        - Yellow (medium %): Model is UNCERTAIN
        - Red (high %): Model thinks it's an ATTACK
        """)

    st.info(f"{num_models} ML models active: {', '.join(model_names)}")

    # Test sample selection
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_idx = st.number_input("Select Test Sample Index (0-10,000)", min_value=0, max_value=10000, value=100, step=1)
    with col2:
        if st.button("Analyze Sample", use_container_width=True):
            X_test, y_test = load_test_data()
            sample = X_test.iloc[sample_idx:sample_idx+1]
            true_label = y_test[sample_idx]

            # X_test is already preprocessed, so pass is_preprocessed=True
            predictions = st.session_state.multi_model.predict_all(sample, is_preprocessed=True)
            agreement = st.session_state.multi_model.get_model_agreement(predictions)

            st.session_state.current_predictions = predictions
            st.session_state.current_agreement = agreement
            st.session_state.current_true_label = true_label

    if hasattr(st.session_state, 'current_predictions'):
        st.markdown("---")

        # Summary metrics with explanations
        col1, col2, col3 = st.columns(3)

        with col1:
            true_is_attack = st.session_state.current_true_label == 1
            st.metric("True Label",
                     "ATTACK" if true_is_attack else "NORMAL",
                     "Ground truth from dataset")

        with col2:
            consensus = st.session_state.current_agreement['consensus'].upper()
            agreement_pct = st.session_state.current_agreement['agreement_percentage']
            st.metric("Consensus",
                     consensus,
                     f"{agreement_pct:.0f}% of models agree")

        with col3:
            attack_votes = st.session_state.current_agreement['attack_votes']
            normal_votes = st.session_state.current_agreement['normal_votes']
            st.metric("Model Votes",
                     f"Attack: {attack_votes} | Normal: {normal_votes}",
                     f"{num_models} total models")

        st.markdown("---")

        # Model predictions table with descriptions
        st.markdown("### Detailed Model Predictions")

        comparison_df = []
        for model_name, result in st.session_state.current_predictions.items():
            display_name = model_name.replace('_', ' ').title()
            prob_pct = result['probability'] * 100

            # Add interpretation
            if result['prediction'] == 'normal':
                interpretation = f"Normal ({100-prob_pct:.1f}% confidence)"
            else:
                interpretation = f"Attack ({prob_pct:.1f}% confidence)"

            comparison_df.append({
                'Model': display_name,
                'Algorithm': model_descriptions.get(display_name, ''),
                'Prediction': result['prediction'].upper(),
                'Attack Probability': f"{result['probability']:.2%}",
                'Confidence': result['confidence'].upper(),
                'Interpretation': interpretation
            })

        st.dataframe(pd.DataFrame(comparison_df), use_container_width=True, height=300)

        st.markdown("---")

        # Probability distribution chart with better explanation
        st.markdown("### Attack Probability Comparison")
        st.caption("Lower values (green) = Model thinks it's normal | Higher values (red) = Model thinks it's an attack")

        probs = [(m.replace('_', ' ').title(), r['probability']) for m, r in st.session_state.current_predictions.items()]
        prob_df = pd.DataFrame(probs, columns=['Model', 'Probability'])

        fig = px.bar(prob_df, x='Model', y='Probability',
                    color='Probability',
                    color_continuous_scale='RdYlGn_r',
                    text='Probability')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(26,31,58,0.5)',
            plot_bgcolor='rgba(26,31,58,0.8)',
            height=450,
            xaxis_title="Machine Learning Model",
            yaxis_title="Attack Probability (%)",
            showlegend=False,
            yaxis_range=[0, min(1.0, max(prob_df['Probability']) * 1.3)],
            coloraxis_colorbar=dict(
                title="Risk Level",
                tickvals=[0, 0.5, 1.0],
                ticktext=["Safe", "Uncertain", "Attack"]
            )
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_anomaly_detection():
    """Tab 3: COMPLETE Anomaly Detection Analysis - ALL 7 MODELS"""
    st.markdown('<div class="section-title">Comprehensive Anomaly Detection Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    **Advanced Anomaly Detection with Multiple Algorithms**
    - Supervised models (Gradient Boosting, XGBoost, LightGBM, Random Forest, Extra Trees, SGD) detect known attack patterns
    - Unsupervised model (Isolation Forest) detects unknown anomalies without labels
    - Ensemble approach combines strengths of all models
    """)

    # Load sample data
    X_test, y_test = load_test_data()

    col1, col2 = st.columns([3, 1])
    with col1:
        sample_size = st.slider("Sample Size for Analysis", 100, 2000, 1000, 100)
    with col2:
        if st.button("Run Complete Analysis", use_container_width=True):
            with st.spinner("Running comprehensive anomaly detection analysis across all 7 models..."):
                try:
                    samples = X_test.iloc[:sample_size]
                    labels = y_test[:sample_size]

                    # X_test is already preprocessed, no need to transform again!
                    X_prep = samples.values  # Convert to numpy array for models
                except Exception as e:
                    st.error(f"Error preparing data: {str(e)}")
                    return

                # Store results for all models
                model_results = {}

                try:
                    for model_name in st.session_state.multi_model.get_available_models():
                        try:
                            model = st.session_state.multi_model.models[model_name]

                            if model_name == 'isolation_forest':
                                # Isolation Forest returns -1 for anomaly, 1 for normal
                                preds = model.predict(X_prep)
                                predictions = (preds == -1).astype(int)
                                # For probability, use decision function scores
                                scores = model.score_samples(X_prep)
                                # Normalize scores to [0,1] range for consistency
                                probabilities = 1 / (1 + np.exp(scores))  # Sigmoid transformation
                            else:
                                # Supervised models
                                predictions = model.predict(X_prep)
                                try:
                                    probabilities = model.predict_proba(X_prep)[:, 1]
                                except:
                                    probabilities = predictions.astype(float)

                            # Calculate metrics
                            cm = confusion_matrix(labels, predictions)

                            # Handle case where confusion matrix doesn't have all 4 values
                            if cm.size == 4:
                                tn, fp, fn, tp = cm.ravel()
                            else:
                                # If only one class predicted, handle gracefully
                                tn = fp = fn = tp = 0
                                if cm.shape[0] == 1:
                                    if labels[0] == 0:
                                        tn = cm[0, 0]
                                    else:
                                        tp = cm[0, 0]

                            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                            # ROC curve
                            try:
                                fpr_curve, tpr_curve, _ = roc_curve(labels, probabilities)
                                roc_auc = auc(fpr_curve, tpr_curve)
                            except:
                                fpr_curve, tpr_curve, roc_auc = [0], [0], 0

                            model_results[model_name] = {
                                'predictions': predictions,
                                'probabilities': probabilities,
                                'confusion_matrix': cm,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'fpr': fpr,
                                'roc_fpr': fpr_curve,
                                'roc_tpr': tpr_curve,
                                'roc_auc': roc_auc,
                                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                            }
                        except Exception as e:
                            st.warning(f"Error processing {model_name}: {str(e)}")
                            continue

                    # Calculate ensemble prediction (majority voting)
                    if len(model_results) > 0:
                        ensemble_votes = np.zeros(len(samples))
                        for result in model_results.values():
                            ensemble_votes += result['predictions']
                        ensemble_preds = (ensemble_votes >= (len(model_results) // 2 + 1)).astype(int)

                        # Ensemble metrics
                        cm_ensemble = confusion_matrix(labels, ensemble_preds)
                        if cm_ensemble.size == 4:
                            tn_e, fp_e, fn_e, tp_e = cm_ensemble.ravel()
                        else:
                            tn_e = fp_e = fn_e = tp_e = 0

                        model_results['ensemble'] = {
                            'predictions': ensemble_preds,
                            'confusion_matrix': cm_ensemble,
                            'accuracy': (tp_e + tn_e) / (tp_e + tn_e + fp_e + fn_e) if (tp_e + tn_e + fp_e + fn_e) > 0 else 0,
                            'precision': tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0,
                            'recall': tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0,
                            'f1': 2 * (tp_e / (tp_e + fp_e)) * (tp_e / (tp_e + fn_e)) / ((tp_e / (tp_e + fp_e)) + (tp_e / (tp_e + fn_e))) if (tp_e + fp_e) > 0 and (tp_e + fn_e) > 0 else 0,
                            'fpr': fp_e / (fp_e + tn_e) if (fp_e + tn_e) > 0 else 0,
                            'tp': tp_e, 'fp': fp_e, 'tn': tn_e, 'fn': fn_e
                        }

                    st.session_state.anomaly_results = model_results
                    st.session_state.anomaly_labels = labels

                    if len(model_results) > 0:
                        st.success(f"Analysis complete. Processed {len(model_results)} models on {sample_size} samples.")
                    else:
                        st.error("No models could process the data successfully.")

                except Exception as e:
                    st.error(f"Critical error during analysis: {str(e)}")
                    st.exception(e)
                    return

    if hasattr(st.session_state, 'anomaly_results'):
        results = st.session_state.anomaly_results

        # Performance Comparison Table
        st.markdown("### Model Performance Comparison")

        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'FPR': result['fpr'],
                'TP': result['tp'],
                'FP': result['fp'],
                'TN': result['tn'],
                'FN': result['fn']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Highlight best performers with proper formatting
        st.dataframe(
            comparison_df.style
                .background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], cmap='Greens')
                .background_gradient(subset=['FPR'], cmap='Reds_r')
                .format({
                    'Accuracy': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}',
                    'FPR': '{:.2%}'
                }),
            use_container_width=True,
            height=350
        )

        st.markdown("---")

        # ROC Curves Comparison
        st.markdown("### ROC Curves - All Models")

        fig = go.Figure()

        colors = ['#00d4ff', '#ff00ea', '#00ff88', '#fbbf24', '#f59e0b', '#dc2626', '#8b5cf6']

        for idx, (model_name, result) in enumerate(results.items()):
            if model_name == 'ensemble':
                continue
            if 'roc_fpr' in result and 'roc_tpr' in result:
                fig.add_trace(go.Scatter(
                    x=result['roc_fpr'],
                    y=result['roc_tpr'],
                    mode='lines',
                    name=f"{model_name.replace('_', ' ').title()} (AUC={result.get('roc_auc', 0):.3f})",
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))

        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title="ROC Curves - True Positive Rate vs False Positive Rate",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            paper_bgcolor='rgba(26,31,58,0.5)',
            plot_bgcolor='rgba(26,31,58,0.8)',
            height=500,
            legend=dict(x=0.6, y=0.1)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Confusion Matrices Grid
        st.markdown("### Confusion Matrices - All Models")

        # Create 3 rows x 3 columns for 8 models (7 + ensemble)
        models_to_show = list(results.keys())

        for row_start in range(0, len(models_to_show), 3):
            cols = st.columns(3)
            for idx, col in enumerate(cols):
                model_idx = row_start + idx
                if model_idx < len(models_to_show):
                    model_name = models_to_show[model_idx]
                    result = results[model_name]

                    with col:
                        cm = result['confusion_matrix']

                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Normal', 'Attack'],
                            y=['Normal', 'Attack'],
                            colorscale='RdYlGn_r',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 16},
                            showscale=False
                        ))

                        fig.update_layout(
                            title=f"{model_name.replace('_', ' ').title()}<br>Acc: {result['accuracy']:.1%}",
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            template="plotly_dark",
                            paper_bgcolor='rgba(26,31,58,0.5)',
                            height=300,
                            margin=dict(l=20, r=20, t=60, b=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Metrics Comparison Charts
        st.markdown("### Metrics Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # Accuracy, Precision, Recall comparison
            metrics_data = []
            for model_name, result in results.items():
                metrics_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': result['accuracy'] * 100,
                    'Precision': result['precision'] * 100,
                    'Recall': result['recall'] * 100
                })

            metrics_df = pd.DataFrame(metrics_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=metrics_df['Model'], y=metrics_df['Accuracy'], marker_color='#00d4ff'))
            fig.add_trace(go.Bar(name='Precision', x=metrics_df['Model'], y=metrics_df['Precision'], marker_color='#00ff88'))
            fig.add_trace(go.Bar(name='Recall', x=metrics_df['Model'], y=metrics_df['Recall'], marker_color='#ff00ea'))

            fig.update_layout(
                title="Accuracy, Precision, Recall Comparison",
                xaxis_title="Model",
                yaxis_title="Percentage (%)",
                template="plotly_dark",
                paper_bgcolor='rgba(26,31,58,0.5)',
                plot_bgcolor='rgba(26,31,58,0.8)',
                height=400,
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # F1-Score and False Positive Rate
            f1_fpr_data = []
            for model_name, result in results.items():
                f1_fpr_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'F1-Score': result['f1'] * 100,
                    'False Positive Rate': result['fpr'] * 100
                })

            f1_fpr_df = pd.DataFrame(f1_fpr_data)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(name='F1-Score', x=f1_fpr_df['Model'], y=f1_fpr_df['F1-Score'], marker_color='#fbbf24'),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(name='False Positive Rate', x=f1_fpr_df['Model'], y=f1_fpr_df['False Positive Rate'],
                          marker_color='#dc2626', mode='lines+markers', line=dict(width=3)),
                secondary_y=True
            )

            fig.update_layout(
                title="F1-Score vs False Positive Rate",
                template="plotly_dark",
                paper_bgcolor='rgba(26,31,58,0.5)',
                plot_bgcolor='rgba(26,31,58,0.8)',
                height=400
            )

            fig.update_xaxes(title_text="Model")
            fig.update_yaxes(title_text="F1-Score (%)", secondary_y=False)
            fig.update_yaxes(title_text="FPR (%)", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Ensemble Performance Highlight
        st.markdown("### Ensemble Performance")

        ensemble_result = results['ensemble']

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Accuracy", f"{ensemble_result['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{ensemble_result['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{ensemble_result['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{ensemble_result['f1']:.2%}")
        with col5:
            st.metric("False Positive Rate", f"{ensemble_result['fpr']:.2%}")

        st.success(f"""
        **Ensemble Advantage:** By combining all 5 models through majority voting, the ensemble achieves
        {ensemble_result['accuracy']:.1%} accuracy with only {ensemble_result['fpr']:.1%} false positive rate.
        This approach leverages the strengths of both supervised and unsupervised learning for robust anomaly detection.
        """)


def tab_network_analysis():
    """Tab 4: Network Traffic Analysis"""
    st.markdown('<div class="section-title">Network Traffic Analysis</div>', unsafe_allow_html=True)

    if len(st.session_state.packet_history) == 0:
        st.info("Start system to collect network data")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Protocol Distribution")
        if st.session_state.protocol_stats:
            proto_df = pd.DataFrame(list(st.session_state.protocol_stats.items()), columns=['Protocol', 'Count'])
            fig = px.pie(proto_df, values='Count', names='Protocol',
                        color_discrete_sequence=px.colors.sequential.Plasma)
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top Services")
        if st.session_state.port_stats:
            port_df = pd.DataFrame(list(st.session_state.port_stats.items()), columns=['Service', 'Count'])
            port_df = port_df.nlargest(10, 'Count')
            fig = px.bar(port_df, x='Service', y='Count', color='Count',
                        color_continuous_scale='Viridis')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)',
                            plot_bgcolor='rgba(26,31,58,0.8)', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### Attack Types")
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

    # ADDED: Dataset Information
    st.markdown("---")
    with st.expander("Dataset information - UNSW-NB15", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Dataset Details:**
            - **Source:** UNSW Canberra Cyber Range Lab
            - **Total Samples:** 82,332 network flows
            - **Features:** 42 network traffic features
            - **Year:** 2015
            """)
        with col2:
            st.markdown("""
            **Attack Categories (9 types):**
            - Backdoor, DoS, Exploits
            - Fuzzers, Generic, Reconnaissance
            - Shellcode, Worms, Analysis
            - **Distribution:** ~56% attacks, ~44% normal
            """)


def main():
    """Main application"""
    render_header()
    render_control_panel()

    st.markdown("---")

    # TABS - Dynamic model count
    num_models = len(st.session_state.multi_model.get_available_models())
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-Time Monitoring",
        f"Model Comparison ({num_models} Models)",
        f"Comprehensive Anomaly Analysis ({num_models} Models + Ensemble)",
        "Network Analysis"
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
