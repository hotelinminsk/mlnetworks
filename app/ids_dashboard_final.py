"""
Streamlit dashboard for live UNSW-NB15 IDS demo
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - must be before other imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from collections import deque
from textwrap import dedent
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.realtime_ids_service import RealtimeIDSService
from services.multi_model_service import MultiModelService
from src.config import DATA_PROCESSED

import warnings
# Suppress the persistent sklearn valid feature names warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---------------------------------------------------------------------------
# Page + global theme
st.set_page_config(
    page_title="Network IDS | Complete Suite",
    page_icon="IDS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        .stApp { background: #0b1120; color: #f8fafc; }
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
            color: #f8fafc;
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
            gap: 8px; background: rgba(30,41,59,0.65);
            padding: 0.4rem; border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(59,130,246,0.15);
            border-radius: 6px; color: #94a3b8;
            font-weight: 600; padding: 0.7rem 1.3rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg,#2563eb 0%,#1d4ed8 100%);
            color: #f8fafc;
        }
        .section-title {
            color: #d1d5db; font-size: 1.4rem; font-weight: 600;
            margin: 1rem 0; padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(148,163,184,0.3);
        }
        .stButton button {
            background: linear-gradient(120deg,#2563eb 0%,#1d4ed8 100%);
            color: #f8fafc; border: none; border-radius: 6px;
            padding: 0.55rem 1.4rem; font-weight: 600;
            box-shadow: 0 12px 20px rgba(37,99,235,0.25);
        }
        div[data-testid="metric-container"] {
            background: rgba(30,41,59,0.85);
            border: 1px solid rgba(148,163,184,0.3);
            border-radius: 12px; padding: 1.2rem 1rem;
            box-shadow: 0 10px 30px rgba(15,23,42,0.35);
        }
        div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.8rem !important; font-weight: 600 !important; }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6rem !important; font-weight: 700 !important; }
        .attack-detail {
            background: rgba(30,41,59,0.9);
            border: 1px solid rgba(248,113,113,0.4);
            border-radius: 12px; padding: 1.5rem;
            margin: 1rem 0; box-shadow: 0 10px 28px rgba(0,0,0,0.35);
        }
        .attack-header { font-size: 1.6rem; font-weight: 700; color: #fca5a5; margin-bottom: 1rem; }
        .attack-info-row {
            background: rgba(15,23,42,0.65);
            padding: 0.9rem 1rem; border-radius: 8px;
            margin: 0.5rem 0; border-left: 4px solid rgba(248,113,113,0.6);
        }
        .attack-label { color: #fda4af; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.08em; }
        .attack-value { color: #f8fafc; font-size: 1rem; font-weight: 500; margin-top: 0.3rem; }
        .traffic-feed {
            background: rgba(15,23,42,0.85);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 12px; overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }
        .traffic-row {
            display: grid; grid-template-columns: 110px 1fr 110px 120px 110px;
            gap: 0.5rem; padding: 0.75rem 1rem; align-items: center;
            font-size: 0.9rem;
        }
        .traffic-row + .traffic-row { border-top: 1px solid rgba(148,163,184,0.15); }
        .traffic-row.header {
            text-transform: uppercase; font-size: 0.7rem;
            letter-spacing: 0.2em; color: #94a3b8;
            background: rgba(15,23,42,0.95);
            border-bottom: 1px solid rgba(148,163,184,0.2);
        }
        .traffic-row.attack { background: rgba(248,113,113,0.08); border-left: 4px solid rgba(248,113,113,0.8); }
        .traffic-row.normal { background: rgba(34,197,94,0.05); border-left: 4px solid rgba(34,197,94,0.7); }
        .traffic-cell { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .traffic-status { font-weight: 700; letter-spacing: 0.05em; }
        .traffic-status.attack { color: #f87171; }
        .traffic-status.normal { color: #34d399; }
        .alert-feed { display: flex; flex-direction: column; gap: 0.6rem; }
        .alert-card {
            padding: 0.9rem 1.1rem; border-radius: 10px;
            background: rgba(15,23,42,0.9);
            border-left: 4px solid #38bdf8;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            color: #e2e8f0;
        }
        .alert-card.critical { border-color: #ef4444; }
        .alert-card.high { border-color: #f97316; }
        .alert-card.medium { border-color: #fbbf24; }
        .alert-card.low { border-color: #22c55e; }
        .alert-card-title { font-weight: 700; margin-bottom: 0.2rem; }
        .alert-card-meta { font-size: 0.75rem; color: #cbd5f5; }
        .model-info-card {
            background: rgba(15,23,42,0.8);
            border: 1px solid rgba(59,130,246,0.4);
            border-radius: 12px; padding: 1rem 1.2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        }
        .model-info-title { font-weight: 700; color: #93c5fd; margin-bottom: 0.6rem; }
        .model-info-body { color: #e2e8f0; line-height: 1.5; }
        .packet-detail {
            background: rgba(15,23,42,0.8);
            border: 1px solid rgba(59,130,246,0.35);
            border-radius: 12px; padding: 1rem 1.2rem;
            margin-bottom: 1rem; box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        }
        .packet-detail h4 { margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #cbd5f5; }
        .packet-row { font-size: 0.9rem; color: #e2e8f0; margin: 0.25rem 0; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        ::-webkit-scrollbar { width: 10px; background: rgba(30,41,59,0.6); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(120deg,#2563eb,#1d4ed8); border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
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
    st.session_state.latest_attack = None
    st.session_state.selected_model = 'gradient_boosting'
    st.session_state.custom_dataset = None
    st.session_state.custom_dataset_name = None
    st.session_state.last_processed = None

# ---------------------------------------------------------------------------
# Data helpers
@st.cache_data
def load_test_data():
    X = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values
    return X, y


def get_dataset_for_evaluation():
    custom_df = st.session_state.get('custom_dataset')
    if custom_df is not None:
        if 'label' not in custom_df.columns:
            name = st.session_state.get('custom_dataset_name', 'Uploaded dataset')
            return None, None, False, name, "Uploaded dataset is unlabeled (missing 'label' column)."
        features = custom_df.drop(columns=['label', 'attack_cat'], errors='ignore')
        labels = custom_df['label'].values
        return features, labels, False, st.session_state.get('custom_dataset_name', 'Uploaded dataset'), None

    X_test, y_test = load_test_data()
    return X_test, y_test, True, "UNSW-NB15 processed test split", None


def evaluate_models_on_dataset(features, labels, sample_size, is_preprocessed):
    if labels is None or len(labels) == 0:
        return [], 0

    indices = np.arange(len(labels))
    if len(indices) > sample_size:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, size=sample_size, replace=False))

    metrics = {}
    for model_name in st.session_state.multi_model.get_available_models():
        metrics[model_name] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'prob_sum': 0.0, 'count': 0}

    for idx in indices:
        row = features.iloc[idx:idx+1] if isinstance(features, pd.DataFrame) else pd.DataFrame(features[idx:idx+1])
        predictions = st.session_state.multi_model.predict_all(row, is_preprocessed=is_preprocessed)
        label = int(labels[idx])

        for model_name, result in predictions.items():
            if result['prediction'] == 'error':
                continue
            pred = 1 if result['prediction'] == 'attack' else 0
            stats = metrics[model_name]
            stats['count'] += 1
            stats['prob_sum'] += result['probability']
            if label == 1 and pred == 1:
                stats['tp'] += 1
            elif label == 0 and pred == 0:
                stats['tn'] += 1
            elif label == 0 and pred == 1:
                stats['fp'] += 1
            else:
                stats['fn'] += 1

    summary = []
    for model_name, stats in metrics.items():
        total = stats['tp'] + stats['tn'] + stats['fp'] + stats['fn']
        if total == 0:
            continue
        accuracy = (stats['tp'] + stats['tn']) / total
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_prob = stats['prob_sum'] / stats['count'] if stats['count'] else 0
        summary.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Avg Attack Probability': avg_prob,
            'Sampled Rows': stats['count']
        })

    return summary, len(indices)

# ---------------------------------------------------------------------------
# UI helper utilities

def _format_timestamp(value, fmt="%H:%M:%S"):
    if isinstance(value, datetime):
        return value.strftime(fmt)
    if value is None:
        return "N/A"
    try:
        return pd.to_datetime(value).strftime(fmt)
    except Exception:
        return str(value)


def render_html_block(html: str):
    cleaned = dedent(html).strip()
    if cleaned:
        st.markdown(cleaned, unsafe_allow_html=True)


def build_packet_row(packet: dict) -> str:
    is_attack = packet.get('prediction') == 'attack'
    status_class = "attack" if is_attack else "normal"
    attack_label = packet.get('attack_type')
    status_label = (attack_label or "ATTACK").upper() if is_attack else "NORMAL"

    duration_value = packet.get('dur')
    try:
        duration_text = f"{float(duration_value):.3f}s"
    except (TypeError, ValueError):
        duration_text = "N/A"

    probability = packet.get('probability')
    probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "N/A"

    row = [
        f'<div class="traffic-row {status_class}">',
        f'<div class="traffic-cell">{_format_timestamp(packet.get("timestamp"))}</div>',
        f'<div class="traffic-cell">{packet.get("service", "-")}</div>',
        f'<div class="traffic-cell">{duration_text}</div>',
        f'<div class="traffic-cell traffic-status {status_class}">{status_label}</div>',
        f'<div class="traffic-cell">{probability_text}</div>',
        '</div>'
    ]
    return "\n".join(row)


def build_alert_card(alert: dict) -> str:
    severity = alert.get('severity', 'low')
    probability = alert.get('probability')
    probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "N/A"
    return dedent(f"""
    <div class="alert-card {severity}">
        <div class="alert-card-title">[{severity.upper()}] {alert.get('attack_type', 'Unknown')}</div>
        <div>{alert.get('message', 'Alert generated')}</div>
        <div class="alert-card-meta">Confidence: {probability_text} | {_format_timestamp(alert.get('timestamp'))}</div>
    </div>
    """)

# Attack explanations
ATTACK_EXPLANATIONS = {
    'Backdoor': {
        'what': 'Backdoor attack – unauthorized remote access attempt',
        'why_dangerous': 'Attacker tries to keep persistent access',
        'indicators': 'Unusual port usage, remote shells',
        'action': 'Block IP, hunt for persistence, forensic scan'
    },
    'Exploits': {
        'what': 'Exploit attempt – abusing software vulnerability',
        'why_dangerous': 'Can escalate privileges or drop payloads',
        'indicators': 'Malformed packets, buffer overflow patterns',
        'action': 'Patch target, isolate asset, review logs'
    },
    'DoS': {
        'what': 'Denial of Service / Flood traffic',
        'why_dangerous': 'Impacts availability, saturates resources',
        'indicators': 'Spike in packets, repeated connection attempts',
        'action': 'Enable rate limiting, block offending IPs'
    },
    'Reconnaissance': {
        'what': 'Recon / scanning',
        'why_dangerous': 'Mapping network before attack',
        'indicators': 'Port sweeps, service enumeration',
        'action': 'Monitor closely, consider blocking source'
    },
    'Shellcode': {
        'what': 'Shellcode injection attempt',
        'why_dangerous': 'Could lead to full compromise',
        'indicators': 'Executable payloads in data field',
        'action': 'Isolate host, run malware analysis'
    },
    'Worms': {
        'what': 'Self-propagating malware',
        'why_dangerous': 'Spreads laterally at speed',
        'indicators': 'Outbound scanning to many hosts',
        'action': 'Quarantine, block egress until cleaned'
    },
    'Fuzzers': {
        'what': 'Fuzzing / malformed traffic tests',
        'why_dangerous': 'Looking for crashes to exploit later',
        'indicators': 'Random payloads, boundary tests',
        'action': 'Activate rate limits, log payloads, patch targets'
    },
    'Analysis': {
        'what': 'Traffic analysis / sniffing',
        'why_dangerous': 'Metadata leakage, recon',
        'indicators': 'Passive taps, unusual sniffing tools',
        'action': 'Enforce encryption, monitor NICs'
    },
    'Generic': {
        'what': 'Generic malicious signal',
        'why_dangerous': 'Unclassified threat pattern',
        'indicators': 'Abnormal protocol mix, heuristics triggered',
        'action': 'Investigate session, capture PCAP'
    }
}


def render_attack_detail(attack_packet: dict):
    attack_type = attack_packet.get('attack_type', 'Unknown')
    explanation = ATTACK_EXPLANATIONS.get(attack_type, ATTACK_EXPLANATIONS['Generic'])
    detection_time = _format_timestamp(attack_packet.get('timestamp'), "%Y-%m-%d %H:%M:%S.%f")
    if detection_time.endswith('N/A'):
        detection_time = _format_timestamp(attack_packet.get('timestamp'))

    duration_value = attack_packet.get('dur')
    try:
        duration_text = f"{float(duration_value):.3f}s"
    except (TypeError, ValueError):
        duration_text = "N/A"

    src_ip = attack_packet.get('srcip', 'N/A')
    dst_ip = attack_packet.get('dstip', 'N/A')
    service = attack_packet.get('service', '-')
    probability = attack_packet.get('probability')
    probability_text = f"{probability:.1%}" if isinstance(probability, (int, float)) else "N/A"

    html_lines = [
        '<div class="attack-detail">',
        f'<div class="attack-header">ATTACK DETECTED: {attack_type.upper()}</div>',
        '<div class="attack-info-row"><div class="attack-label">WHEN</div>'
        f'<div class="attack-value">{detection_time}</div></div>',
        '<div class="attack-info-row"><div class="attack-label">WHAT</div>'
        f'<div class="attack-value">{explanation["what"]}</div></div>',
        '<div class="attack-info-row"><div class="attack-label">WHY DANGEROUS</div>'
        f'<div class="attack-value">{explanation["why_dangerous"]}</div></div>',
        '<div class="attack-info-row"><div class="attack-label">INDICATORS</div>'
        f'<div class="attack-value">{explanation["indicators"]}</div></div>',
        '<div class="attack-info-row"><div class="attack-label">SOURCE & TARGET</div>'
        f'<div class="attack-value">From: <span style="color:#ef4444;">{src_ip}</span> -&gt; '
        f'To: <span style="color:#fbbf24;">{dst_ip}</span> | '
        f'Service: <span style="color:#38bdf8;">{service}</span> | '
        f'Duration: <span style="color:#93c5fd;">{duration_text}</span></div></div>',
        '<div class="attack-info-row"><div class="attack-label">MODEL CONFIDENCE</div>'
        f'<div class="attack-value">{probability_text}</div></div>',
        '<div class="attack-info-row"><div class="attack-label">RECOMMENDED ACTION</div>'
        f'<div class="attack-value" style="color:#fca5a5; font-weight:800;">{explanation["action"]}</div></div>',
        '</div>'
    ]
    render_html_block("\n".join(html_lines))


def render_current_packet_detail():
    packet = st.session_state.get('last_processed')
    if not packet:
        return

    raw_packet = packet.get('raw_packet') or {}
    row_idx = raw_packet.get('replay_index', 'N/A')
    dataset_label = st.session_state.get('custom_dataset_name') or 'Default UNSW-NB15 testing set'

    prediction = packet.get('prediction', 'unknown').upper()
    probability = packet.get('probability', 0)
    label = packet.get('true_label')
    if label == 1:
        actual_text = 'ATTACK (labelled)'
    elif label == 0:
        actual_text = 'NORMAL (labelled)'
    else:
        actual_text = 'Unknown / unlabeled'

    summary_keys = ['srcip', 'dstip', 'service', 'proto', 'state', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
    summary = []
    for key in summary_keys:
        value = raw_packet.get(key)
        if value is not None:
            summary.append(f"{key}: {value}")
    summary_text = ", ".join(summary[:6]) if summary else "No raw columns available"

    preview_lines = []
    for key, value in raw_packet.items():
        preview_lines.append(f"{key}: {value}")
        if len(preview_lines) >= 12:
            break

    st.markdown(f"""
    <div class="packet-detail">
        <h4>Row #{row_idx} from {dataset_label}</h4>
        <div class="packet-row"><strong>Row data:</strong> {summary_text}</div>
        <div class="packet-row"><strong>Prediction:</strong> {prediction} ({probability:.1%})</div>
        <div class="packet-row"><strong>Actual label:</strong> {actual_text}</div>
        <div class="packet-row"><strong>Timestamp:</strong> {_format_timestamp(packet.get('timestamp'))}</div>
    </div>
    """, unsafe_allow_html=True)
    if preview_lines:
        st.code("\n".join(preview_lines), language="text")


# ---------------------------------------------------------------------------
# Header & controls

def render_header():
    st.markdown('<div class="mega-header">Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Threat Detection & Analysis - UNSW-NB15 Dataset</div>', unsafe_allow_html=True)


def render_control_panel():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Start System", use_container_width=True, disabled=st.session_state.is_running):
            model_to_use = st.session_state.get('selected_model', 'gradient_boosting')
            data_frame = st.session_state.get('custom_dataset')
            st.session_state.ids_service = RealtimeIDSService(model_name=model_to_use, data_frame=data_frame)
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
            if st.session_state.ids_service:
                st.session_state.ids_service.stop()
                st.session_state.ids_service.reset()
            st.session_state.is_running = False
            st.session_state.packet_history.clear()
            st.session_state.alert_history.clear()
            st.session_state.model_predictions.clear()
            st.session_state.latest_attack = None
            st.session_state.protocol_stats.clear()
            st.session_state.port_stats.clear()
            st.session_state.last_processed = None
            if hasattr(st.session_state, 'anomaly_results'):
                delattr(st.session_state, 'anomaly_results')
            if hasattr(st.session_state, 'current_predictions'):
                delattr(st.session_state, 'current_predictions')
            if hasattr(st.session_state, 'current_agreement'):
                delattr(st.session_state, 'current_agreement')
            if hasattr(st.session_state, 'model_eval_results'):
                delattr(st.session_state, 'model_eval_results')
            st.session_state.ids_service = None
            st.session_state.multi_model = MultiModelService()
            st.rerun()

    with col4:
        status = "ACTIVE" if st.session_state.is_running else "INACTIVE"
        icon = "●" if st.session_state.is_running else "○"
        color = '#10b981' if st.session_state.is_running else '#ef4444'
        st.markdown(f"<div style='text-align:center;padding:0.5rem;font-size:1.3rem;font-weight:700;color:{color};'>{icon} {status}</div>", unsafe_allow_html=True)

    info_model = st.session_state.get('selected_model', 'gradient_boosting').replace('_', ' ').title()
    info_dataset = st.session_state.get('custom_dataset_name', 'Default UNSW-NB15 testing set')
    if st.session_state.is_running and st.session_state.ids_service:
        info_model = st.session_state.ids_service.model_name.replace('_', ' ').title()
        info_dataset = st.session_state.get('custom_dataset_name', 'Default UNSW-NB15 testing set') \
            if st.session_state.get('custom_dataset') is not None else "Default UNSW-NB15 testing set"
        card_title = "Currently analyzing"
    else:
        card_title = "Next run will use"

    st.markdown(f"""
        <div class="model-info-card" style="margin-top:0.5rem">
            <div class="model-info-title">{card_title}</div>
            <div class="model-info-body">
                <strong>Model:</strong> {info_model}<br>
                <strong>Data source:</strong> {info_dataset}
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
# Tabs implementation

def tab_realtime_monitoring():
    st.markdown('<div class="section-title">Real-Time Network Monitoring</div>', unsafe_allow_html=True)

    if not st.session_state.ids_service or not st.session_state.is_running:
        st.info("Click Start System to begin real-time monitoring.")
        st.markdown("---")
        st.markdown('<div class="section-title" style="margin-top:0;">System Configuration</div>', unsafe_allow_html=True)

        available_models = st.session_state.multi_model.get_available_models()
        current_selection = st.session_state.get('selected_model', 'gradient_boosting')
        current_index = available_models.index(current_selection) if current_selection in available_models else 0

        col1, col2 = st.columns([1.2, 1])
        with col1:
            selected_model = st.selectbox(
                "Select Detection Model",
                options=available_models,
                format_func=lambda x: x.replace('_', ' ').title(),
                index=current_index,
                key='model_selector'
            )
            if st.button("Apply Model", use_container_width=True):
                st.session_state.selected_model = selected_model
                st.session_state.ids_service = None
                st.success(f"Model selection updated to {selected_model.replace('_', ' ').title()}. Start the system to apply it.")

        with col2:
            dataset_label = st.session_state.get('custom_dataset_name') or "Default UNSW-NB15 testing set"
            
            # Create styled badges for models - compact to single line to avoid markdown code blocks
            badges_html = ""
            for m in available_models:
                display_name = m.replace('_', ' ').title()
                is_selected = (m == selected_model)
                bg_color = "rgba(59,130,246,0.5)" if is_selected else "rgba(30,41,59,0.5)"
                border_color = "#60a5fa" if is_selected else "rgba(148,163,184,0.3)"
                text_color = "#ffffff" if is_selected else "#94a3b8"
                weight = "700" if is_selected else "500"
                
                # Use single line string to prevent markdown indentation issues
                badges_html += f'<span style="display:inline-block;padding:4px 10px;margin:2px;background:{bg_color};border:1px solid {border_color};border-radius:12px;color:{text_color};font-size:0.75rem;font-weight:{weight};">{display_name}</span>'

            # Styled Data Source Badge - compact to single line
            ds_badge = f'<span style="display:inline-block;padding:3px 10px;background:rgba(124,58,237,0.15);border:1px solid rgba(139,92,246,0.3);border-radius:10px;color:#ddd6fe;font-size:0.8rem;font-weight:600;">{dataset_label}</span>'

            # Render final HTML block
            st.markdown(dedent(f"""
                <div class="model-info-card">
                    <div class="model-info-title">Pending Model: {selected_model.replace('_', ' ').title()}</div>
                    <div class="model-info-body">
                        <div style="margin-bottom:12px;display:flex;align-items:center;gap:8px;">
                            <span style="color:#94a3b8;font-weight:600;font-size:0.85rem;">DATA SOURCE:</span>
                            {ds_badge}
                        </div>
                        <div style="color:#94a3b8;font-weight:600;font-size:0.85rem;margin-bottom:6px;">AVAILABLE MODELS:</div>
                        <div style="display:flex;flex-wrap:wrap;gap:4px;">
                            {badges_html}
                        </div>
                    </div>
                </div>
            """).strip(), unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV or Parquet for live replay", type=["csv", "parquet"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith('.parquet'):
                    custom_df = pd.read_parquet(uploaded_file)
                else:
                    custom_df = pd.read_csv(uploaded_file)
                st.session_state.custom_dataset = custom_df
                st.session_state.custom_dataset_name = uploaded_file.name
                st.session_state.ids_service = None
                st.success(f"Loaded {len(custom_df):,} records from {uploaded_file.name}. Start the system to replay this dataset.")
            except Exception as exc:
                st.error(f"Failed to load file: {exc}")

        if st.session_state.custom_dataset is not None:
            if st.button("Remove Uploaded Dataset", use_container_width=True):
                st.session_state.custom_dataset = None
                st.session_state.custom_dataset_name = None
                st.success("Custom dataset removed. Default UNSW dataset will be used.")

        active_dataset_label = st.session_state.get('custom_dataset_name', 'Default UNSW-NB15 testing set')
        st.markdown(
            f"<div style='margin-top:0.5rem;font-size:0.9rem;color:#94a3b8;'>"
            f"<strong>When you press Start System</strong> the dashboard will stream data from {active_dataset_label} using the selected model above."
            f"</div>",
            unsafe_allow_html=True
        )
        return

    # When running ---------------------------------------------------------
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
                st.session_state.ids_service = RealtimeIDSService(
                    model_name=selected_model,
                    data_frame=st.session_state.get('custom_dataset')
                )
                st.session_state.ids_service.start()
                st.session_state.selected_model = selected_model
                st.success(f"Switched to: {selected_model.replace('_', ' ').title()}")
                st.rerun()
        with col2:
            st.info(f"**Current:** {current_model.replace('_', ' ').title()}")

    stats = st.session_state.ids_service.get_statistics()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Packets", f"{stats['packets_processed']:,}", f"+{len(st.session_state.packet_history)}")
    with col2:
        st.metric("Threats", f"{stats['attacks_detected']:,}", f"{stats['alert_summary']['critical']} Critical")
    with col3:
        st.metric("Benign", f"{stats['normal_traffic']:,}", "Verified")
    with col4:
        st.metric("Accuracy", f"{stats.get('accuracy', 0):.1f}%", f"{stats['true_positives']} TP")
    with col5:
        attack_rate = (stats['attacks_detected'] / stats['packets_processed'] * 100) if stats['packets_processed'] else 0
        st.metric("Threat Rate", f"{attack_rate:.1f}%", "Live")

    render_current_packet_detail()
    st.markdown("---")

    # Process 1 packet per rerun for smoother "row by row" feel
    for _ in range(1):
        packet = st.session_state.ids_service.process_next_packet()
        if not packet:
            continue
        st.session_state.packet_history.append(packet)
        st.session_state.last_processed = packet
        if packet['prediction'] == 'attack' and packet.get('attack_type') and packet['attack_type'] != 'Normal':
            st.session_state.latest_attack = packet
        # Get protocol from raw_packet if available, otherwise from packet
        raw_packet = packet.get('raw_packet', {})
        proto = raw_packet.get('proto') or packet.get('proto') or 'Unknown'
        if proto and str(proto).strip() and str(proto) != 'nan':
            st.session_state.protocol_stats[str(proto)] = st.session_state.protocol_stats.get(str(proto), 0) + 1
        
        # Get service from raw_packet if available, otherwise from packet
        service = raw_packet.get('service') or packet.get('service') or 'Unknown'
        if service and str(service).strip() and str(service) != 'nan' and str(service) != '-':
            st.session_state.port_stats[str(service)] = st.session_state.port_stats.get(str(service), 0) + 1

    if st.session_state.latest_attack:
        st.markdown("### Latest Attack Detected - Full Analysis")
        render_attack_detail(st.session_state.latest_attack)
        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Live Traffic Stream")
        recent = list(st.session_state.packet_history)[-15:]
        if recent:
            rows_html = "".join(build_packet_row(pkt) for pkt in reversed(recent))
            render_html_block(f"""
            <div class="traffic-feed">
                <div class="traffic-row header">
                    <div class="traffic-cell">Time</div>
                    <div class="traffic-cell">Service</div>
                    <div class="traffic-cell">Duration</div>
                    <div class="traffic-cell">Status</div>
                    <div class="traffic-cell">Confidence</div>
                </div>
                {rows_html}
            </div>
            """)
        else:
            st.info("Waiting for packets...")

    with col2:
        st.markdown("### Security Alerts")
        alerts = st.session_state.ids_service.get_recent_alerts(10)
        if alerts:
            cards_html = "".join(build_alert_card(alert) for alert in reversed(alerts))
            render_html_block(f"<div class=\"alert-feed\">{cards_html}</div>")
        else:
            st.success("No threats detected")

def tab_model_comparison():
    num_models = len(st.session_state.multi_model.get_available_models())
    st.markdown(f'<div class="section-title">Multi-Model Comparison ({num_models} Models)</div>', unsafe_allow_html=True)

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
        - **Prediction**: ATTACK or NORMAL decision per model
        - **Attack Probability**: Likelihood this is an attack (0% = normal, 100% = attack)
        - **Confidence**: HIGH when probability is <30% or >70%

        **Model consensus**
        - Agreement % shows how aligned the models are

        **Chart colors**
        - Green = normal, Yellow = uncertain, Red = attack
        """)

    st.info(f"{num_models} ML models active: {', '.join(model_names)}")

    col1, col2 = st.columns([3, 1])
    with col1:
        sample_idx = st.number_input("Select Test Sample Index (0-10,000)", min_value=0, max_value=10000, value=100, step=1)
    with col2:
        if st.button("Analyze Sample", use_container_width=True):
            X_test, y_test = load_test_data()
            sample = X_test.iloc[sample_idx:sample_idx+1]
            true_label = y_test[sample_idx]
            predictions = st.session_state.multi_model.predict_all(sample, is_preprocessed=True)
            agreement = st.session_state.multi_model.get_model_agreement(predictions)
            st.session_state.current_predictions = predictions
            st.session_state.current_agreement = agreement
            st.session_state.current_true_label = true_label

    if hasattr(st.session_state, 'current_predictions'):
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            true_is_attack = st.session_state.current_true_label == 1
            st.metric("True Label", "ATTACK" if true_is_attack else "NORMAL", "Ground truth")
        with col2:
            agreement = st.session_state.current_agreement
            st.metric("Consensus", agreement['consensus'].upper(), f"{agreement['agreement_percentage']:.0f}% agree")
        with col3:
            st.metric("Model Votes", f"Attack: {agreement['attack_votes']} | Normal: {agreement['normal_votes']}", f"{num_models} total")

        st.markdown("### Detailed Model Predictions")
        comparison_data = []
        for model_name, result in st.session_state.current_predictions.items():
            display_name = model_name.replace('_', ' ').title()
            prob_pct = result['probability'] * 100
            interpretation = f"Attack ({prob_pct:.1f}% confidence)" if result['prediction'] == 'attack' else f"Normal ({100-prob_pct:.1f}% confidence)"
            comparison_data.append({
                'Model': display_name,
                'Algorithm': model_descriptions.get(display_name, ''),
                'Prediction': result['prediction'].upper(),
                'Attack Probability': f"{result['probability']:.2%}",
                'Confidence': result['confidence'].upper(),
                'Interpretation': interpretation
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, height=300, key="model_predictions_table")

        st.markdown("### Attack Probability Comparison")
        st.caption("Lower values (green) = model thinks it's normal | Higher values (red) = model thinks it's an attack")
        prob_df = pd.DataFrame([(m.replace('_', ' ').title(), r['probability']) for m, r in st.session_state.current_predictions.items()], columns=['Model', 'Probability'])
        fig = px.bar(prob_df, x='Model', y='Probability', color='Probability', color_continuous_scale='RdYlGn_r', text='Probability')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)', plot_bgcolor='rgba(26,31,58,0.8)', height=450,
                          xaxis_title="Model", yaxis_title="Attack Probability", showlegend=False,
                          yaxis_range=[0, min(1.0, max(prob_df['Probability'].max(), 0.2))],
                          coloraxis_colorbar=dict(title="Risk", tickvals=[0, 0.5, 1.0], ticktext=["Safe", "Uncertain", "Attack"]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Dataset Evaluation (Aggregated Metrics)")
    features, labels, is_preprocessed, dataset_name, eval_warning = get_dataset_for_evaluation()
    if eval_warning:
        st.warning(eval_warning)
    else:
        total_rows = len(labels)
        if total_rows == 0:
            st.info("No labeled rows available for evaluation.")
        else:
            default_value = min(total_rows, 300)
            sample_size = st.slider("Rows to evaluate", min_value=1, max_value=total_rows, value=default_value, step=1, key="dataset_eval_sample")
            if st.button("Run Dataset Evaluation", use_container_width=True):
                with st.spinner(f"Evaluating models on {sample_size} rows from {dataset_name}..."):
                    summary, evaluated = evaluate_models_on_dataset(features, labels, sample_size, is_preprocessed)
                    st.session_state.model_eval_results = {'summary': summary, 'dataset_name': dataset_name, 'sample_size': evaluated}

    if 'model_eval_results' in st.session_state and st.session_state.model_eval_results.get('summary'):
        eval_result = st.session_state.model_eval_results
        summary_df = pd.DataFrame(eval_result['summary'])
        if summary_df.empty:
            st.info("Evaluation ran but returned no metrics.")
        else:
            st.success(f"Evaluated {len(summary_df)} models on {eval_result['sample_size']} rows from {eval_result['dataset_name']}.")
            styled = summary_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1 Score': '{:.2%}',
                'Avg Attack Probability': '{:.2%}'
            }).highlight_max(subset=['Accuracy', 'F1 Score'], color='#14532d')
            st.dataframe(styled, use_container_width=True, height=350, key="model_eval_table")
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
            with st.spinner("Running comprehensive anomaly detection analysis across all models..."):
                try:
                    # Use same random sampling approach as tab 2 (evaluate_models_on_dataset)
                    # This ensures consistent results between tabs
                    indices = np.arange(len(y_test))
                    if len(indices) > sample_size:
                        rng = np.random.default_rng(42)  # Same seed as tab 2
                        indices = np.sort(rng.choice(indices, size=sample_size, replace=False))
                    else:
                        indices = indices
                    
                    samples = X_test.iloc[indices]
                    labels = y_test[indices]

                    # X_test is already preprocessed (from DATA_PROCESSED), so use it directly
                    # But we need to ensure it's in the right format for models
                    X_prep = samples.values  # Convert to numpy array for models
                    
                    # Verify data shape
                    if X_prep.shape[0] != len(labels):
                        st.error(f"Data shape mismatch: features {X_prep.shape[0]} vs labels {len(labels)}")
                        return
                        
                except Exception as e:
                    st.error(f"Error preparing data: {str(e)}")
                    st.exception(e)
                    return

                # Store results for all models - use same approach as tab 2 (evaluate_models_on_dataset)
                # Process each sample individually using predict_all() for consistency
                model_results = {}
                
                # Initialize metrics for all models
                for model_name in st.session_state.multi_model.get_available_models():
                    model_results[model_name] = {
                        'predictions': [],
                        'probabilities': [],
                        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
                    }

                try:
                    # Process each sample individually using predict_all() - same as tab 2
                    progress_bar = st.progress(0)
                    for idx in range(len(samples)):
                        row = samples.iloc[idx:idx+1]  # Get single row as DataFrame
                        true_label = int(labels[idx])
                        
                        # Use predict_all() with is_preprocessed=True - same as tab 2
                        predictions = st.session_state.multi_model.predict_all(row, is_preprocessed=True)
                        
                        for model_name, result in predictions.items():
                            if result['prediction'] == 'error':
                                continue
                            
                            pred = 1 if result['prediction'] == 'attack' else 0
                            prob = result['probability']
                            
                            model_results[model_name]['predictions'].append(pred)
                            model_results[model_name]['probabilities'].append(prob)
                            
                            # Calculate confusion matrix components
                            if true_label == 1 and pred == 1:
                                model_results[model_name]['tp'] += 1
                            elif true_label == 0 and pred == 0:
                                model_results[model_name]['tn'] += 1
                            elif true_label == 0 and pred == 1:
                                model_results[model_name]['fp'] += 1
                            else:
                                model_results[model_name]['fn'] += 1
                        
                        # Update progress
                        if (idx + 1) % 100 == 0:
                            progress_bar.progress((idx + 1) / len(samples))
                    
                    progress_bar.progress(1.0)
                    
                    # Calculate metrics for each model
                    for model_name in st.session_state.multi_model.get_available_models():
                        if model_name not in model_results:
                            continue
                        
                        result = model_results[model_name]
                        tp, tn, fp, fn = result['tp'], result['tn'], result['fp'], result['fn']
                        
                        # Calculate metrics
                        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        
                        # Convert lists to numpy arrays for ROC calculation
                        predictions_array = np.array(result['predictions'])
                        probabilities_array = np.array(result['probabilities'])
                        
                        # ROC curve
                        try:
                            if len(np.unique(probabilities_array)) > 1 and not np.isnan(probabilities_array).any():
                                fpr_curve, tpr_curve, _ = roc_curve(labels, probabilities_array)
                                roc_auc = auc(fpr_curve, tpr_curve)
                            else:
                                fpr_curve, tpr_curve = np.array([0, 1]), np.array([0, 1])
                                roc_auc = 0.5
                        except Exception as e:
                            fpr_curve, tpr_curve = np.array([0, 1]), np.array([0, 1])
                            roc_auc = 0.5
                        
                        # Update model_results with calculated metrics
                        cm = np.array([[tn, fp], [fn, tp]])
                        model_results[model_name].update({
                            'confusion_matrix': cm,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'fpr': fpr,
                            'roc_fpr': fpr_curve,
                            'roc_tpr': tpr_curve,
                            'roc_auc': roc_auc,
                            'predictions': predictions_array,
                            'probabilities': probabilities_array
                        })
                    
                    # Remove any models that failed to process
                    model_results = {k: v for k, v in model_results.items() if 'accuracy' in v}

                    # Calculate ensemble prediction (majority voting)
                    # Exclude 'ensemble' key if it exists
                    if len(model_results) > 0:
                        ensemble_votes = np.zeros(len(samples))
                        model_count = 0
                        for model_name, result in model_results.items():
                            if model_name != 'ensemble':  # Exclude ensemble itself
                                # Ensure predictions is a numpy array
                                preds = np.array(result['predictions']) if 'predictions' in result else np.zeros(len(samples))
                                ensemble_votes += preds
                                model_count += 1
                        
                        # Majority voting: if more than half vote for attack (1), predict attack
                        threshold = model_count / 2.0
                        ensemble_preds = (ensemble_votes > threshold).astype(int)

                        # Ensemble metrics
                        cm_ensemble = confusion_matrix(labels, ensemble_preds)
                        if cm_ensemble.size == 4:
                            tn_e, fp_e, fn_e, tp_e = cm_ensemble.ravel()
                        else:
                            tn_e = fp_e = fn_e = tp_e = 0

                        # Calculate ensemble metrics correctly
                        precision_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0
                        recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
                        f1_e = 2 * (precision_e * recall_e) / (precision_e + recall_e) if (precision_e + recall_e) > 0 else 0
                        
                        # Calculate ensemble probabilities (average of all model probabilities)
                        ensemble_probs = np.zeros(len(samples))
                        prob_count = 0
                        for model_name, result in model_results.items():
                            if model_name != 'ensemble' and 'probabilities' in result:
                                # Ensure probabilities is a numpy array
                                probs = np.array(result['probabilities'])
                                ensemble_probs += probs
                                prob_count += 1
                        
                        if prob_count > 0:
                            ensemble_probs = ensemble_probs / prob_count
                        else:
                            ensemble_probs = ensemble_preds.astype(float)
                        
                        # Calculate ROC curve for ensemble
                        try:
                            if len(np.unique(ensemble_probs)) > 1 and not np.isnan(ensemble_probs).any():
                                fpr_curve_e, tpr_curve_e, _ = roc_curve(labels, ensemble_probs)
                                roc_auc_e = auc(fpr_curve_e, tpr_curve_e)
                            else:
                                fpr_curve_e, tpr_curve_e = np.array([0, 1]), np.array([0, 1])
                                roc_auc_e = 0.5
                        except:
                            fpr_curve_e, tpr_curve_e = np.array([0, 1]), np.array([0, 1])
                            roc_auc_e = 0.5
                        
                        model_results['ensemble'] = {
                            'predictions': ensemble_preds,
                            'probabilities': ensemble_probs,
                            'confusion_matrix': cm_ensemble,
                            'accuracy': (tp_e + tn_e) / (tp_e + tn_e + fp_e + fn_e) if (tp_e + tn_e + fp_e + fn_e) > 0 else 0,
                            'precision': precision_e,
                            'recall': recall_e,
                            'f1': f1_e,
                            'fpr': fp_e / (fp_e + tn_e) if (fp_e + tn_e) > 0 else 0,
                            'roc_fpr': fpr_curve_e,
                            'roc_tpr': tpr_curve_e,
                            'roc_auc': roc_auc_e,
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

        model_idx = 0
        for model_name, result in results.items():
            if 'roc_fpr' in result and 'roc_tpr' in result:
                # Check if ROC data is valid (not just [0, 1] default)
                fpr_data = result['roc_fpr']
                tpr_data = result['roc_tpr']
                
                # Skip if invalid (all zeros or default values)
                if isinstance(fpr_data, np.ndarray) and len(fpr_data) > 2:
                    # Valid ROC curve
                    display_name = model_name.replace('_', ' ').title()
                    if model_name == 'ensemble':
                        display_name = 'Ensemble (Majority Voting)'
                    
                    fig.add_trace(go.Scatter(
                        x=fpr_data,
                        y=tpr_data,
                        mode='lines',
                        name=f"{display_name} (AUC={result.get('roc_auc', 0):.3f})",
                        line=dict(color=colors[model_idx % len(colors)], width=2 if model_name != 'ensemble' else 3)
                    ))
                    model_idx += 1

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
    st.markdown('<div class="section-title">Network Traffic Analysis</div>', unsafe_allow_html=True)

    st.checkbox("Pause auto-refresh (fullscreen için etkinleştirin)", key="pause_refresh")

    total_packets = len(st.session_state.packet_history)
    attack_packets = len([p for p in st.session_state.packet_history if p['prediction'] == 'attack'])
    attack_pct = (attack_packets / total_packets * 100) if total_packets else 0
    dataset_label = st.session_state.get('custom_dataset_name', 'Default UNSW-NB15 testing set')

    st.markdown(f"""
    <div class="model-info-card" style="margin-bottom:1rem;">
        <div class="model-info-title">Live dataset insights</div>
        <div class="model-info-body">
            <strong>Source file:</strong> {dataset_label}<br>
            <strong>Packets analyzed:</strong> {total_packets:,} (attacks: {attack_packets:,}, {attack_pct:.1f}% of stream)<br>
            <strong>Replay speed:</strong> slowed (~2 packets / refresh) for operator visibility.<br>
            <em>Charts update as additional packets flow through the system.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if total_packets == 0:
        st.info("Start system to collect network data")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Protocol Distribution")
        if st.session_state.protocol_stats:
            proto_df = pd.DataFrame(list(st.session_state.protocol_stats.items()), columns=['Protocol', 'Count'])
            # Filter out 'Unknown' if it's the only value or if there are other values
            if len(proto_df) > 1:
                proto_df = proto_df[proto_df['Protocol'] != 'Unknown']
            if len(proto_df) > 0:
                fig = px.pie(proto_df, values='Count', names='Protocol', color_discrete_sequence=px.colors.sequential.Plasma)
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)', height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No protocol data available yet")
        else:
            st.info("No protocol data collected yet")

    with col2:
        st.markdown("### Top Services")
        if st.session_state.port_stats:
            port_df = pd.DataFrame(list(st.session_state.port_stats.items()), columns=['Service', 'Count']).nlargest(10, 'Count')
            fig = px.bar(port_df, x='Service', y='Count', color='Count', color_continuous_scale='Viridis')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)', plot_bgcolor='rgba(26,31,58,0.8)', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### Attack Types")
        attack_packets = [p for p in st.session_state.packet_history if p['prediction'] == 'attack' and p.get('attack_type')]
        if attack_packets:
            attack_types = [p['attack_type'] for p in attack_packets]
            attack_dist = pd.Series(attack_types).value_counts()
            fig = px.bar(x=attack_dist.index, y=attack_dist.values, color=attack_dist.values, color_continuous_scale='Reds')
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(26,31,58,0.5)', plot_bgcolor='rgba(26,31,58,0.8)', height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"*Most common attack so far:* **{attack_dist.index[0]}** ({attack_dist.iloc[0]} packets)")

    st.markdown("---")
    with st.expander("Dataset information - UNSW-NB15", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Dataset Details:**
            - Source: UNSW Canberra Cyber Range Lab
            - 82,332 flow records, 42 features
            - Capture year: 2015
            """)
        with col2:
            st.markdown("""
            **Attack Categories:**
            - Backdoor, DoS, Exploits
            - Fuzzers, Generic, Reconnaissance
            - Shellcode, Worms, Analysis
            - Rough split: ~56% attack / ~44% normal
            """)


def main():
    render_header()
    render_control_panel()
    st.markdown("---")

    # Auto-refresh control (useful when opening charts fullscreen)
    if 'pause_refresh' not in st.session_state:
        st.session_state.pause_refresh = False

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

    if st.session_state.is_running and not st.session_state.pause_refresh:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
