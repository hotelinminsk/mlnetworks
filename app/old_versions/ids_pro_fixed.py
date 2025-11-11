"""
PROFESSIONAL IDS - Detailed Attack Analysis
Shows WHAT, WHEN, HOW, WHY for every attack
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from pathlib import Path
import sys
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.realtime_ids_service import RealtimeIDSService
from services.multi_model_service import MultiModelService
from src.config import DATA_PROCESSED

st.set_page_config(
    page_title="Professional IDS | Attack Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS - FIXED PADDING
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }

    /* FIX: Proper padding */
    .main .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }

    .mega-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff 0%, #ff00ea 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 0.5rem 0;
        text-shadow: 0 0 30px rgba(0,212,255,0.6);
    }

    .sub-title {
        text-align: center;
        color: #8b92b0;
        font-size: 1.3rem;
        margin-bottom: 2rem;
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
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    /* Attack Detail Panel */
    .attack-detail {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(220,38,38,0.2) 100%);
        border: 2px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: pulse-border 2s infinite;
    }

    @keyframes pulse-border {
        0%, 100% { border-color: #dc2626; box-shadow: 0 0 10px rgba(220,38,38,0.3); }
        50% { border-color: #ef4444; box-shadow: 0 0 25px rgba(239,68,68,0.6); }
    }

    .attack-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #fca5a5;
        margin-bottom: 1rem;
    }

    .attack-info-row {
        background: rgba(26,31,58,0.6);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ef4444;
    }

    .attack-label {
        color: #fca5a5;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
    }

    .attack-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Normal traffic */
    .normal-traffic {
        background: rgba(16,185,129,0.1);
        border-left: 3px solid #10b981;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        color: #6ee7b7;
    }

    /* Section title */
    .section-title {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid rgba(0,212,255,0.4);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize
if 'ids_service' not in st.session_state:
    st.session_state.ids_service = None
    st.session_state.multi_model = MultiModelService()
    st.session_state.is_running = False
    st.session_state.packet_history = deque(maxlen=300)
    st.session_state.attack_details = deque(maxlen=50)
    st.session_state.latest_attack = None


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
    """Render detailed attack information"""
    attack_type = attack_packet.get('attack_type', 'Unknown')
    explanation = get_attack_explanation(attack_type)

    st.markdown(f"""
    <div class="attack-detail">
        <div class="attack-header">
            🚨 ATTACK DETECTED: {attack_type.upper()}
        </div>

        <div class="attack-info-row">
            <div class="attack-label">⏰ WHEN (Detection Time)</div>
            <div class="attack-value">{attack_packet['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}</div>
        </div>

        <div class="attack-info-row">
            <div class="attack-label">🎯 WHAT (Attack Type)</div>
            <div class="attack-value">{explanation['what']}</div>
        </div>

        <div class="attack-info-row">
            <div class="attack-label">⚠️ WHY DANGEROUS</div>
            <div class="attack-value">{explanation['why_dangerous']}</div>
        </div>

        <div class="attack-info-row">
            <div class="attack-label">🔍 HOW DETECTED (Indicators)</div>
            <div class="attack-value">{explanation['indicators']}</div>
        </div>

        <div class="attack-info-row">
            <div class="attack-label">📍 SOURCE & TARGET</div>
            <div class="attack-value">
                From: <span style="color: #ef4444;">{attack_packet['srcip']}</span> →
                To: <span style="color: #fbbf24;">{attack_packet['dstip']}</span> |
                Service: <span style="color: #00d4ff;">{attack_packet['service']}</span>
            </div>
        </div>

        <div class="attack-info-row">
            <div class="attack-label">🎯 ML MODEL CONFIDENCE</div>
            <div class="attack-value">{attack_packet['probability']:.1%} - Model is {attack_packet['probability']*100:.0f}% confident this is an attack</div>
        </div>

        <div class="attack-info-row">
            <div class="attack-label">✅ RECOMMENDED ACTION</div>
            <div class="attack-value" style="color: #fca5a5; font-weight: 800;">{explanation['action']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="mega-header">🛡️ PROFESSIONAL NETWORK IDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Attack Detection & Analysis | ML-Powered Threat Intelligence</div>', unsafe_allow_html=True)

    # Control panel
    col1, col2, col3, col4 = st.columns([2,2,2,2])

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
            st.session_state.attack_details.clear()
            st.session_state.latest_attack = None
            st.rerun()

    with col4:
        status = "🟢 ACTIVE" if st.session_state.is_running else "🔴 INACTIVE"
        st.markdown(f"<div style='text-align: center; padding: 0.7rem; font-size: 1.5rem; font-weight: 800; color: {'#10b981' if st.session_state.is_running else '#ef4444'};'>{status}</div>", unsafe_allow_html=True)

    st.markdown("---")

    if not st.session_state.is_running:
        st.info("🚀 Click START SYSTEM to begin real-time attack detection and analysis")
        return

    # KPIs
    stats = st.session_state.ids_service.get_statistics()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📡 PACKETS", f"{stats['packets_processed']:,}")
    with col2:
        st.metric("🚨 ATTACKS", f"{stats['attacks_detected']:,}", f"{stats['alert_summary']['critical']} Critical")
    with col3:
        st.metric("✅ NORMAL", f"{stats['normal_traffic']:,}")
    with col4:
        accuracy = stats.get('accuracy', 0)
        st.metric("🎯 ACCURACY", f"{accuracy:.1f}%")
    with col5:
        attack_rate = (stats['attacks_detected'] / stats['packets_processed'] * 100) if stats['packets_processed'] > 0 else 0
        st.metric("⚡ ATTACK RATE", f"{attack_rate:.1f}%")

    st.markdown("---")

    # Process packets
    for _ in range(8):
        packet = st.session_state.ids_service.process_next_packet()
        if packet:
            st.session_state.packet_history.append(packet)

            if packet['prediction'] == 'attack' and packet.get('attack_type') and packet['attack_type'] != 'Normal':
                st.session_state.latest_attack = packet
                st.session_state.attack_details.append(packet)

    # LATEST ATTACK DETAIL
    if st.session_state.latest_attack:
        st.markdown('<div class="section-title">🚨 LATEST ATTACK DETECTED</div>', unsafe_allow_html=True)
        render_attack_detail(st.session_state.latest_attack)

    st.markdown("---")

    # Two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">🔴 LIVE TRAFFIC FEED</div>', unsafe_allow_html=True)

        recent = list(st.session_state.packet_history)[-15:]
        for pkt in reversed(recent):
            is_attack = pkt['prediction'] == 'attack'

            if is_attack and pkt.get('attack_type') and pkt['attack_type'] != 'Normal':
                st.markdown(f"""
                <div class="attack-detail" style="padding: 1rem; margin: 0.3rem 0;">
                    🚨 <strong style="color: #fca5a5;">{pkt['timestamp'].strftime('%H:%M:%S')}</strong> |
                    <strong style="color: #ef4444;">{pkt['attack_type']}</strong> |
                    {pkt['srcip']} → {pkt['dstip']} |
                    {pkt['service']} |
                    <strong>{pkt['probability']:.1%}</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-traffic">
                    ✅ {pkt['timestamp'].strftime('%H:%M:%S')} |
                    {pkt['srcip']} → {pkt['dstip']} |
                    {pkt['service']}
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">📊 ATTACK SUMMARY</div>', unsafe_allow_html=True)

        if len(st.session_state.attack_details) > 0:
            attack_types = [a['attack_type'] for a in st.session_state.attack_details if a.get('attack_type')]
            attack_counts = pd.Series(attack_types).value_counts()

            st.markdown(f"**Total Attacks:** {len(st.session_state.attack_details)}")
            st.markdown("**Attack Types:**")

            for attack, count in attack_counts.items():
                st.markdown(f"- 🔴 **{attack}**: {count} times")
        else:
            st.success("✅ No attacks detected yet")

    # Auto-refresh
    if st.session_state.is_running:
        time.sleep(0.4)
        st.rerun()


if __name__ == "__main__":
    main()
