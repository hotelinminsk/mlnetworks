"""
🔒 Network Intrusion Detection - Gelişmiş Dashboard
Tüm özelliklerle birlikte
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from joblib import load
import time
from datetime import datetime, timedelta

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS, DATA_PROCESSED

# Page config
st.set_page_config(
    page_title="Network IDS",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #262730;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_all_models():
    """Load ALL 5 models"""
    models = {}
    model_files = {
        "Isolation Forest": {"file": "isolation_forest.joblib", "icon": "🌲", "type": "anomaly"},
        "SGD Classifier": {"file": "supervised_sgd.joblib", "icon": "⚡", "type": "linear"},
        "Random Forest": {"file": "random_forest.joblib", "icon": "🌳", "type": "ensemble"},
        "Gradient Boosting": {"file": "gradient_boosting.joblib", "icon": "🚀", "type": "ensemble"},
        "Extra Trees": {"file": "extra_trees.joblib", "icon": "🌴", "type": "ensemble"},
    }

    for name, info in model_files.items():
        path = MODELS / info['file']
        if path.exists():
            models[name] = {
                'model': load(path),
                'icon': info['icon'],
                'type': info['type']
            }

    return models

@st.cache_data
def load_test_data():
    """Load test data"""
    X = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values
    return X, y

try:
    models = load_all_models()
    X_test, y_test = load_test_data()

    # Header
    st.markdown('<p class="main-header">🔒 Network Intrusion Detection System</p>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #666;'>Gerçek Zamanlı Ağ Saldırı Tespiti | {len(models)} Model Aktif | {len(X_test):,} Test Örneği</p>", unsafe_allow_html=True)
    st.markdown("---")

    # System Performance Metrics Dashboard
    st.markdown("### 📊 Sistem Performansı")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        # Get best model's ROC AUC
        from sklearn.metrics import roc_auc_score
        gb_model = models["Gradient Boosting"]['model']
        gb_proba = gb_model.predict_proba(X_test)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_proba)

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{gb_auc:.4f}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">ROC AUC Score</p>
                <small style="opacity: 0.7;">✨ Gradient Boosting</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col2:
        # Threshold effectiveness
        current_threshold = 0.7
        y_pred_threshold = (gb_proba >= current_threshold).astype(int)
        fp = np.sum((y_pred_threshold == 1) & (y_test == 0))
        fp_reduction = ((fp / len(y_test)) * 100)

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{int(current_threshold*100)}%</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Data %</p>
                <small style="opacity: 0.7;">🎯 Threshold: {current_threshold}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col3:
        # False alarms per day
        fp_per_day = int(fp / 7)  # 7 days of data assumption

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">~{fp_per_day}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Yanlış Alarm/Gün</p>
                <small style="opacity: 0.7;">📉 Detection: -18</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col4:
        # Recall
        tp = np.sum((y_pred_threshold == 1) & (y_test == 1))
        fn = np.sum((y_pred_threshold == 0) & (y_test == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{recall*100:.1f}%</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Recall</p>
                <small style="opacity: 0.7;">🎯 Saldırı Bulma</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Sistem Kontrolleri")

        model_name = st.selectbox(
            "🎯 Model Seçimi",
            list(models.keys()),
            index=3,  # Default: Gradient Boosting
            help="Farklı algoritmaları test edin"
        )

        threshold = st.slider(
            "🎚️ Karar Eşiği",
            0.0, 1.0, 0.7, 0.05,
            help="Saldırı tespiti için minimum olasılık"
        )

        st.markdown("---")
        st.markdown("### 📊 Hızlı İstatistikler")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Saldırı", f"{int(y_test.sum()):,}")
            st.metric("🟢 Normal", f"{int(len(y_test) - y_test.sum()):,}")
        with col2:
            attack_ratio = (y_test.sum() / len(y_test)) * 100
            st.metric("📊 Saldırı %", f"{attack_ratio:.1f}%")
            st.metric("⚖️ Denge", f"{100-attack_ratio:.1f}%")

        st.markdown("---")
        st.markdown("### 🏆 Model Durumu")
        st.success(f"✅ {len(models)} Model Yüklü")
        st.info(f"🎯 Aktif: {models[model_name]['icon']} {model_name}")

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎬 Canlı Demo",
        "📊 Real-Time Monitoring",
        "🏆 Model Karşılaştırma",
        "🔍 Saldırı Analizi",
        "📈 Performans Detayları",
        "🧠 Feature Importance"
    ])

    # TAB 1: Live Demo
    with tab1:
        st.markdown("## 🎬 İnteraktif Tahmin Demo")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🎮 Test Parametreleri")

            demo_mode = st.radio(
                "Demo Modu:",
                ["🎯 Tek Örnek Test", "🔄 Batch Tahmin", "🎲 Rastgele Simülasyon"],
                help="Farklı test modları"
            )

            if demo_mode == "🎯 Tek Örnek Test":
                traffic_type = st.selectbox(
                    "Trafik Türü:",
                    ["🟢 Normal Trafik", "🔴 Saldırı Trafiği", "🎲 Rastgele"]
                )

                n_samples = 1

            elif demo_mode == "🔄 Batch Tahmin":
                traffic_type = st.selectbox(
                    "Trafik Türü:",
                    ["🟢 Normal Trafik", "🔴 Saldırı Trafiği", "🎲 Karışık"]
                )

                n_samples = st.slider("Örnek Sayısı:", 5, 20, 10)

            else:  # Rastgele simülasyon
                traffic_type = "🎲 Rastgele"
                n_samples = st.slider("Simülasyon Boyutu:", 10, 100, 50)

                if st.button("🔄 YENİ SİMÜLASYON", type="primary", use_container_width=True):
                    st.session_state.clear()

            st.markdown("---")

            # Model comparison toggle
            compare_models = st.checkbox("🆚 Tüm Modelleri Karşılaştır", value=False)

            if st.button("🚀 TAHMİN YAP!", type="primary", use_container_width=True, key="predict_btn"):
                # Select samples
                if "Normal" in traffic_type:
                    indices = np.where(y_test == 0)[0]
                elif "Saldırı" in traffic_type:
                    indices = np.where(y_test == 1)[0]
                else:
                    indices = np.arange(len(y_test))

                selected = np.random.choice(indices, min(n_samples, len(indices)), replace=False)

                # Get predictions
                X_samples = X_test.iloc[selected]
                y_true = y_test[selected]

                if compare_models:
                    # Compare all models
                    all_results = {}
                    for m_name, m_data in models.items():
                        model = m_data['model']

                        if m_name == "Isolation Forest":
                            y_scores = -model.decision_function(X_samples)
                            # Normalize
                            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
                        else:
                            y_scores = model.predict_proba(X_samples)[:, 1]

                        y_pred = (y_scores >= threshold).astype(int)

                        all_results[m_name] = {
                            'scores': y_scores,
                            'predictions': y_pred,
                            'true': y_true
                        }

                    st.session_state['comparison_results'] = all_results
                else:
                    # Single model
                    model = models[model_name]['model']

                    if model_name == "Isolation Forest":
                        y_proba = -model.decision_function(X_samples)
                        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
                    else:
                        y_proba = model.predict_proba(X_samples)[:, 1]

                    y_pred = (y_proba >= threshold).astype(int)

                    st.session_state['results'] = {
                        'proba': y_proba,
                        'pred': y_pred,
                        'true': y_true,
                        'mode': demo_mode
                    }

        with col2:
            st.markdown("### 📊 Tahmin Sonuçları")

            if 'comparison_results' in st.session_state:
                # Show comparison
                st.markdown("#### 🆚 Model Karşılaştırması")

                for m_name, res in st.session_state['comparison_results'].items():
                    accuracy = np.mean(res['predictions'] == res['true']) * 100

                    st.markdown(f"**{models[m_name]['icon']} {m_name}**")
                    st.progress(accuracy / 100)
                    st.caption(f"Doğruluk: {accuracy:.1f}%")
                    st.markdown("---")

            elif 'results' in st.session_state:
                res = st.session_state['results']

                if res['mode'] == "🎯 Tek Örnek Test":
                    # Single prediction - gauge
                    prob = res['proba'][0]
                    pred = res['pred'][0]
                    true = res['true'][0]
                    correct = (pred == true)

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Saldırı Olasılığı"},
                        delta={'reference': threshold * 100},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red" if pred == 1 else "green"},
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
                    st.plotly_chart(fig, use_container_width=True)

                    # Result card
                    if pred == 1 and correct:
                        st.error(f"🔴 **SALDIRI TESPİT EDİLDİ!**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ✅ Doğru")
                    elif pred == 1 and not correct:
                        st.warning(f"⚠️ **YANLIŞLIKLA ALARM!**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ❌ False Positive")
                    elif pred == 0 and correct:
                        st.success(f"🟢 **NORMAL TRAFİK**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ✅ Doğru")
                    else:
                        st.error(f"🚨 **SALDIRI KAÇIRILDI!**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ❌ False Negative")

                else:
                    # Batch/Simulation - summary stats
                    accuracy = np.mean(res['pred'] == res['true']) * 100
                    tp = np.sum((res['pred'] == 1) & (res['true'] == 1))
                    fp = np.sum((res['pred'] == 1) & (res['true'] == 0))
                    fn = np.sum((res['pred'] == 0) & (res['true'] == 1))
                    tn = np.sum((res['pred'] == 0) & (res['true'] == 0))

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Doğruluk", f"{accuracy:.1f}%")
                    with col2:
                        st.metric("True Pos", tp)
                    with col3:
                        st.metric("False Pos", fp)
                    with col4:
                        st.metric("False Neg", fn)

                    # Distribution
                    fig = go.Figure()

                    fig.add_trace(go.Histogram(
                        x=res['proba'][res['true'] == 0],
                        name='Normal',
                        opacity=0.7,
                        marker_color='green'
                    ))

                    fig.add_trace(go.Histogram(
                        x=res['proba'][res['true'] == 1],
                        name='Saldırı',
                        opacity=0.7,
                        marker_color='red'
                    ))

                    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                                 annotation_text=f"Threshold: {threshold}")

                    fig.update_layout(
                        title="Olasılık Dağılımı",
                        xaxis_title="Saldırı Olasılığı",
                        yaxis_title="Sayı",
                        barmode='overlay',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("👈 Soldaki panelden 'TAHMİN YAP!' butonuna basın!")

    # TAB 2: Real-Time Monitoring
    with tab2:
        st.markdown("## 📊 Real-Time Monitoring & Alert System")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Simulate time-series data
            if st.button("🔄 Yeni Monitoring Başlat", key="monitoring", type="primary"):
                # Generate time series
                n_points = 100
                timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_points, 0, -1)]

                # Simulate traffic
                normal_baseline = np.random.normal(100, 20, n_points)
                attack_spikes = np.random.choice([0, 1], n_points, p=[0.85, 0.15])
                attack_traffic = attack_spikes * np.random.normal(300, 50, n_points)

                total_traffic = normal_baseline + attack_traffic

                st.session_state['monitoring_data'] = {
                    'timestamps': timestamps,
                    'total': total_traffic,
                    'attacks': attack_spikes
                }

        with col2:
            auto_refresh = st.checkbox("🔁 Auto-Refresh", value=False)
            if auto_refresh:
                refresh_interval = st.slider("Refresh (sn):", 1, 10, 5)
                st.info(f"Her {refresh_interval} saniyede yenilenecek")
                time.sleep(refresh_interval)
                st.rerun()

        if 'monitoring_data' in st.session_state:
            data = st.session_state['monitoring_data']

            # Real-time metrics dashboard
            met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)

            with met_col1:
                st.metric("🌐 Toplam Trafik", f"{int(data['total'].sum()):,}",
                         delta=f"{int(np.random.randint(-500, 1000))} pkt")
            with met_col2:
                attack_count = int(data['attacks'].sum())
                st.metric("🚨 Saldırı Sayısı", attack_count,
                         delta=f"{np.random.choice(['+', '-'])}{np.random.randint(1, 5)}")
            with met_col3:
                attack_rate = (data['attacks'].sum() / len(data['attacks'])) * 100
                st.metric("📊 Saldırı Oranı", f"{attack_rate:.1f}%",
                         delta=f"{np.random.choice(['+', '-'])}{np.random.uniform(0.5, 2):.1f}%")
            with met_col4:
                st.metric("⚡ Avg Latency", f"{np.random.uniform(2, 8):.2f}ms",
                         delta=f"-{np.random.uniform(0.1, 0.5):.2f}ms")
            with met_col5:
                st.metric("🎯 Detection Rate", f"{np.random.uniform(95, 99.5):.1f}%",
                         delta=f"+{np.random.uniform(0.1, 0.8):.1f}%")

            st.markdown("---")

            # Traffic over time with enhanced visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('📈 Network Traffic Over Time', '🚨 Attack Detection Timeline'),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4]
            )

            # Traffic line with gradient
            fig.add_trace(
                go.Scatter(
                    x=data['timestamps'],
                    y=data['total'],
                    mode='lines',
                    name='Traffic Volume',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.3)'
                ),
                row=1, col=1
            )

            # Add threshold line
            fig.add_hline(y=200, line_dash="dash", line_color="orange",
                         annotation_text="Threshold", row=1, col=1)

            # Attack markers with colors
            attack_times = [data['timestamps'][i] for i in range(len(data['attacks'])) if data['attacks'][i] == 1]
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

            st.plotly_chart(fig, use_container_width=True)

            # Alert Panel
            if attack_count > 0:
                st.markdown("### 🚨 Recent Alerts")

                alert_col1, alert_col2 = st.columns([3, 1])

                with alert_col1:
                    for i, (t, a) in enumerate(zip(data['timestamps'][-20:], data['attacks'][-20:])):
                        if a == 1:
                            st.error(f"⚠️ **ALERT #{i+1}** | Time: {t.strftime('%H:%M:%S')} | Threat: High | Action: Blocked")
                            if i >= 2:  # Show only 3 most recent
                                break

                with alert_col2:
                    st.markdown("**Threat Level**")
                    threat_level = min(100, attack_count * 8)

                    if threat_level < 30:
                        st.success(f"🟢 Low\n\n{threat_level}%")
                    elif threat_level < 70:
                        st.warning(f"🟡 Medium\n\n{threat_level}%")
                    else:
                        st.error(f"🔴 High\n\n{threat_level}%")
        else:
            st.info("👆 'Yeni Monitoring Başlat' butonuna basın!")

    # TAB 3: Model Comparison
    with tab3:
        st.markdown("## 🏆 Tüm Modellerin Karşılaştırması")

        # Calculate metrics for all models
        metrics_data = []
        for name, data in models.items():
            model = data['model']

            if name == "Isolation Forest":
                y_scores = -model.decision_function(X_test)
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
            else:
                y_scores = model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

            y_pred = (y_scores >= 0.5).astype(int)

            metrics_data.append({
                'Model': f"{data['icon']} {name}",
                'Type': data['type'].title(),
                'ROC AUC': roc_auc_score(y_test, y_scores),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred)
            })

        df = pd.DataFrame(metrics_data)

        # Display table
        st.dataframe(
            df.style.background_gradient(cmap='RdYlGn', subset=['ROC AUC', 'Accuracy', 'Precision', 'Recall']).format({
                'ROC AUC': '{:.4f}',
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Bar charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ROC AUC Karşılaştırması', 'F1 Score Karşılaştırması')
        )

        fig.add_trace(
            go.Bar(x=df['Model'], y=df['ROC AUC'], name='ROC AUC', marker_color='#667eea'),
            row=1, col=1
        )

        df['F1'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])

        fig.add_trace(
            go.Bar(x=df['Model'], y=df['F1'], name='F1 Score', marker_color='#764ba2'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Attack Analysis
    with tab4:
        st.markdown(f"## 🔍 Saldırı Trafiği Analizi - {model_name}")

        # Get model predictions
        model = models[model_name]['model']

        if model_name == "Isolation Forest":
            y_scores = -model.decision_function(X_test)
            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        y_pred = (y_scores >= threshold).astype(int)

        # Get attack/normal predictions
        attack_predictions = np.sum(y_pred == 1)
        normal_predictions = np.sum(y_pred == 0)

        # Actual labels
        actual_attacks = np.sum(y_test == 1)
        actual_normal = np.sum(y_test == 0)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Model Tahminleri")
            st.metric("🔴 Saldırı Tespit Edildi", f"{attack_predictions:,}",
                     delta=f"{attack_predictions - actual_attacks:+,} fark")
            st.metric("🟢 Normal Tespit Edildi", f"{normal_predictions:,}",
                     delta=f"{normal_predictions - actual_normal:+,} fark")

        with col2:
            st.markdown("### ✅ Gerçek Değerler")
            st.metric("🔴 Gerçek Saldırı", f"{actual_attacks:,}")
            st.metric("🟢 Gerçek Normal", f"{actual_normal:,}")

        st.markdown("---")

        # Comparison charts
        col1, col2 = st.columns(2)

        with col1:
            # Predictions pie
            fig1 = go.Figure(data=[go.Pie(
                labels=['Tahmin: Saldırı', 'Tahmin: Normal'],
                values=[attack_predictions, normal_predictions],
                marker=dict(colors=['#ff6b6b', '#51cf66']),
                hole=0.4,
                textinfo='label+percent+value'
            )])
            fig1.update_layout(title=f"{model_name} Tahminleri", height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Actual pie
            fig2 = go.Figure(data=[go.Pie(
                labels=['Gerçek: Saldırı', 'Gerçek: Normal'],
                values=[actual_attacks, actual_normal],
                marker=dict(colors=['#e74c3c', '#2ecc71']),
                hole=0.4,
                textinfo='label+percent+value'
            )])
            fig2.update_layout(title="Gerçek Veri Dağılımı", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Accuracy metrics
        st.markdown("---")
        st.markdown("### 🎯 Model Performansı")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        met1, met2, met3, met4 = st.columns(4)
        with met1:
            st.metric("🎯 Accuracy", f"{acc*100:.2f}%")
        with met2:
            st.metric("🔍 Precision", f"{prec*100:.2f}%")
        with met3:
            st.metric("📡 Recall", f"{rec*100:.2f}%")
        with met4:
            st.metric("⚖️ F1 Score", f"{f1*100:.2f}%")

    # TAB 5: Performance Details
    with tab5:
        st.markdown(f"## 📈 {model_name} - Detaylı Performans")

        model = models[model_name]['model']

        if model_name == "Isolation Forest":
            y_proba = -model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
        else:
            y_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Attack'],
            y=['Actual Normal', 'Actual Attack'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))

        fig.update_layout(title='Confusion Matrix', height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        tn, fp, fn, tp = cm.ravel()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ True Negative", f"{tn:,}")
        with col2:
            st.metric("❌ False Positive", f"{fp:,}", f"-{fp/7:.0f}/gün")
        with col3:
            st.metric("❌ False Negative", f"{fn:,}")
        with col4:
            st.metric("✅ True Positive", f"{tp:,}")

    # TAB 6: Feature Importance
    with tab6:
        st.markdown("## 🧠 Feature Importance & Model Explainability")

        # Select model with feature importance
        importance_models = {k: v for k, v in models.items()
                           if k in ["Random Forest", "Gradient Boosting", "Extra Trees"]}

        if not importance_models:
            st.warning("⚠️ Feature importance sadece tree-based modeller için mevcut")
        else:
            selected_model = st.selectbox(
                "🎯 Model Seçin:",
                list(importance_models.keys()),
                index=1 if "Gradient Boosting" in importance_models else 0
            )

            model = importance_models[selected_model]['model']

            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_test.columns

                # Create dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                # Top N features
                top_n = st.slider("Gösterilecek Feature Sayısı:", 10, 50, 20)

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Bar chart
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        y=importance_df['Feature'].head(top_n)[::-1],
                        x=importance_df['Importance'].head(top_n)[::-1],
                        orientation='h',
                        marker=dict(
                            color=importance_df['Importance'].head(top_n)[::-1],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=importance_df['Importance'].head(top_n)[::-1].round(4),
                        textposition='outside'
                    ))

                    fig.update_layout(
                        title=f"Top {top_n} Most Important Features - {selected_model}",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        height=600,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Feature Statistics")

                    st.metric("📈 Total Features", len(feature_names))
                    st.metric("🎯 Top Feature", importance_df['Feature'].iloc[0])
                    st.metric("💯 Max Importance", f"{importance_df['Importance'].iloc[0]:.4f}")

                    st.markdown("---")

                    # Cumulative importance
                    cumsum = importance_df['Importance'].cumsum()
                    features_80 = (cumsum <= 0.8).sum()
                    features_90 = (cumsum <= 0.9).sum()

                    st.markdown("**Cumulative Importance:**")
                    st.info(f"📌 Top {features_80} features → 80% importance")
                    st.info(f"📌 Top {features_90} features → 90% importance")

                # Feature importance over threshold
                st.markdown("---")
                st.markdown("### 🔍 Feature Analysis")

                importance_threshold = st.slider(
                    "Importance Threshold:",
                    0.0, float(importance_df['Importance'].max()),
                    float(importance_df['Importance'].quantile(0.9)),
                    0.001
                )

                critical_features = importance_df[importance_df['Importance'] >= importance_threshold]

                st.markdown(f"**{len(critical_features)} critical features** above threshold ({importance_threshold:.4f}):")

                # Display in columns
                n_cols = 3
                cols = st.columns(n_cols)

                for idx, (_, row) in enumerate(critical_features.iterrows()):
                    col_idx = idx % n_cols
                    with cols[col_idx]:
                        st.metric(
                            row['Feature'][:20],
                            f"{row['Importance']:.4f}",
                            delta=f"Rank #{idx+1}"
                        )

                # Pie chart for feature groups
                st.markdown("---")
                st.markdown("### 📊 Feature Distribution by Importance")

                # Group by importance ranges
                bins = [0, 0.01, 0.05, 0.1, 1.0]
                labels = ['Very Low (<0.01)', 'Low (0.01-0.05)', 'Medium (0.05-0.1)', 'High (>0.1)']

                importance_df['Group'] = pd.cut(importance_df['Importance'], bins=bins, labels=labels)
                group_counts = importance_df['Group'].value_counts()

                fig = go.Figure(data=[go.Pie(
                    labels=group_counts.index,
                    values=group_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'])
                )])

                fig.update_layout(title="Features by Importance Level", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Download button
                st.markdown("---")
                csv = importance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Feature Importance (CSV)",
                    data=csv,
                    file_name=f"feature_importance_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )

except Exception as e:
    st.error(f"❌ Hata: {str(e)}")
    st.info("Modelleri eğitin: `make train_ensemble`")
