"""
Network Intrusion Detection System - Main Dashboard
Clean Code & SOLID Principles:
- Single Responsibility: Her modül tek bir sorumluluğa sahip
- Open/Closed: Genişletmeye açık, değişikliğe kapalı
- Dependency Inversion: Service layer kullanarak bağımlılıkları azaltıyoruz
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import time
from typing import Dict, Any
import streamlit.components.v1 as components

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import services and components
from config import (
    PAGE_CONFIG, DEFAULT_THRESHOLD, DEFAULT_MODEL_INDEX,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    MIN_SAMPLES, MAX_SAMPLES, DEFAULT_BATCH_SAMPLES,
    MIN_SIMULATION, MAX_SIMULATION, DEFAULT_SIMULATION,
    MIN_FEATURES, MAX_FEATURES, DEFAULT_TOP_N_FEATURES,
    MONITORING_POINTS, REFRESH_INTERVAL_MIN, REFRESH_INTERVAL_MAX, DEFAULT_REFRESH_INTERVAL
)
from services.model_service import ModelService
from services.metrics_service import MetricsService
from services.monitoring_service import MonitoringService
from components.metrics_display import MetricsDisplay
from components.chart_components import ChartComponents
from src.config import DATA_PROCESSED

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Lucide Icons - Load once globally
st.markdown("""
<script src="https://unpkg.com/lucide@latest"></script>
<script>
    // Create icons on page load and after any DOM updates
    document.addEventListener('DOMContentLoaded', () => lucide.createIcons());
    
    // Re-create icons after Streamlit reruns
    const observer = new MutationObserver(() => {
        lucide.createIcons();
    });
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    /* Lucide icon styling */
    .lucide {
        display: inline-block;
        vertical-align: middle;
        stroke: currentColor;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    
    .main-header .lucide {
        width: 50px;
        height: 50px;
        stroke: #667eea;
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


@st.cache_resource
def load_test_data():
    """Test verisini yükle"""
    X = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values
    return X, y


def render_header(models: Dict[str, Any], X_test: pd.DataFrame) -> None:
    """Header bölümünü render et"""
    st.markdown(
        '<div class="main-header"><i data-lucide="shield"></i>Network Intrusion Detection System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center; color: #666;'>"
        f"Gerçek Zamanlı Ağ Saldırı Tespiti | {len(models)} Model Aktif | {len(X_test):,} Test Örneği</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_performance_metrics(
    model_service: ModelService,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float
) -> None:
    """Performans metriklerini render et"""
    st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.3rem;font-weight:600;"><i data-lucide="trending-up" style="width:24px;height:24px;"></i><span>Sistem Performansı</span></div>', unsafe_allow_html=True)
    
    # Get model predictions with error handling
    try:
        y_proba = model_service.predict_proba(model_name, X_test)
        y_pred = model_service.predict(model_name, X_test, threshold)
    except Exception as e:
        st.error(f"Model tahmin hatası: {str(e)}")
        st.info("Lütfen farklı bir model seçin veya modelleri yeniden eğitin.")
        return
    
    # Calculate metrics
    metrics = MetricsService.calculate_all_metrics(y_test, y_pred, y_proba)
    
    # Display metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{metrics['roc_auc']:.4f}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">ROC AUC Score</p>
                <small style="opacity: 0.7;">{model_name}</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with metric_col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{int(threshold*100)}%</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Threshold</p>
                <small style="opacity: 0.7;">Decision Threshold</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with metric_col3:
        fp_per_day = int(metrics['fp'] / 7)  # 7 days assumption
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">~{fp_per_day}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Yanlış Alarm/Gün</p>
                <small style="opacity: 0.7;">False Positives</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with metric_col4:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                        padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2.5rem;">{metrics['recall']*100:.1f}%</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Recall</p>
                <small style="opacity: 0.7;">Saldırı Bulma</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")


def render_sidebar(
    models: Dict[str, Any],
    y_test: np.ndarray,
    model_name: str
) -> str:
    """Sidebar'ı render et ve seçilen model adını döndür"""
    with st.sidebar:
        st.markdown('<div style="display:flex;align-items:center;gap:10px;font-size:1.3rem;font-weight:600;"><i data-lucide="settings" style="width:24px;height:24px;"></i><span>Sistem Kontrolleri</span></div>', unsafe_allow_html=True)
        
        model_name = st.selectbox(
            "Model Seçimi",
            list(models.keys()),
            index=DEFAULT_MODEL_INDEX if DEFAULT_MODEL_INDEX < len(models) else 0,
            help="Farklı algoritmaları test edin"
        )
        
        threshold = st.slider(
            "Karar Eşiği",
            THRESHOLD_MIN, THRESHOLD_MAX, DEFAULT_THRESHOLD, THRESHOLD_STEP,
            help="Saldırı tespiti için minimum olasılık"
        )
        
        st.markdown("---")
        st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.1rem;font-weight:600;"><i data-lucide="bar-chart-2" style="width:20px;height:20px;"></i><span>Hızlı İstatistikler</span></div>', unsafe_allow_html=True)
        
        MetricsDisplay.display_summary_stats(y_test)
        
        st.markdown("---")
        st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.1rem;font-weight:600;"><i data-lucide="award" style="width:20px;height:20px;"></i><span>Model Durumu</span></div>', unsafe_allow_html=True)
        st.success(f"✓ {len(models)} Model Yüklü")
        
        # Display active model with its icon
        icon_html = f'<i data-lucide="{models[model_name]["icon"]}" style="width:16px;height:16px;"></i>'
        st.markdown(f'<div style="display:flex;align-items:center;gap:6px;padding:8px;background:rgba(99,102,241,0.1);border-radius:6px;">{icon_html}<span>Aktif: {model_name}</span></div>', unsafe_allow_html=True)
        
        return model_name, threshold


def render_live_demo_tab(
    model_service: ModelService,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float
) -> None:
    """Canlı demo tab'ını render et"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.3rem;font-weight:600;"><i data-lucide="sliders" style="width:24px;height:24px;"></i><span>Test Parametreleri</span></div>', unsafe_allow_html=True)
        
        demo_mode = st.radio(
            "Demo Modu:",
            ["Tek Örnek Test", "Batch Tahmin", "Rastgele Simülasyon"],
            help="Farklı test modları"
        )
        
        if demo_mode == "Tek Örnek Test":
            traffic_type = st.selectbox(
                "Trafik Türü:",
                ["Normal Trafik", "Saldırı Trafiği", "Rastgele"]
            )
            n_samples = 1
        elif demo_mode == "Batch Tahmin":
            traffic_type = st.selectbox(
                "Trafik Türü:",
                ["Normal Trafik", "Saldırı Trafiği", "Karışık"]
            )
            n_samples = st.slider("Örnek Sayısı:", MIN_SAMPLES, MAX_SAMPLES, DEFAULT_BATCH_SAMPLES)
        else:  # Rastgele simülasyon
            traffic_type = "Rastgele"
            n_samples = st.slider("Simülasyon Boyutu:", MIN_SIMULATION, MAX_SIMULATION, DEFAULT_SIMULATION)
            
            if st.button("YENİ SİMÜLASYON", type="primary", use_container_width=True):
                st.session_state.clear()
        
        st.markdown("---")
        compare_models = st.checkbox("Tüm Modelleri Karşılaştır", value=False)
        
        if st.button("TAHMİN YAP!", type="primary", use_container_width=True, key="predict_btn"):
            # Select samples
            if "Normal" in traffic_type:
                indices = np.where(y_test == 0)[0]
            elif "Saldırı" in traffic_type:
                indices = np.where(y_test == 1)[0]
            else:
                indices = np.arange(len(y_test))
            
            selected = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
            X_samples = X_test.iloc[selected]
            y_true = y_test[selected]
            
            if compare_models:
                # Compare all models
                all_results = {}
                for m_name in model_service._models.keys():
                    y_scores = model_service.predict_proba(m_name, X_samples)
                    y_pred = model_service.predict(m_name, X_samples, threshold)
                    all_results[m_name] = {
                        'scores': y_scores,
                        'predictions': y_pred,
                        'true': y_true
                    }
                st.session_state['comparison_results'] = all_results
            else:
                # Single model
                y_proba = model_service.predict_proba(model_name, X_samples)
                y_pred = model_service.predict(model_name, X_samples, threshold)
                
                st.session_state['results'] = {
                    'proba': y_proba,
                    'pred': y_pred,
                    'true': y_true,
                    'mode': demo_mode
                }
    
    with col2:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.3rem;font-weight:600;"><i data-lucide="bar-chart-3" style="width:24px;height:24px;"></i><span>Tahmin Sonuçları</span></div>', unsafe_allow_html=True)
        
        if 'comparison_results' in st.session_state:
            st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.1rem;font-weight:600;"><i data-lucide="git-compare" style="width:20px;height:20px;"></i><span>Model Karşılaştırması</span></div>', unsafe_allow_html=True)
            for m_name, res in st.session_state['comparison_results'].items():
                accuracy = np.mean(res['predictions'] == res['true']) * 100
                st.markdown(f"**{model_service._models[m_name]['icon']} {m_name}**")
                st.progress(accuracy / 100)
                st.caption(f"Doğruluk: {accuracy:.1f}%")
                st.markdown("---")
        
        elif 'results' in st.session_state:
            res = st.session_state['results']
            
            if res['mode'] == "Tek Örnek Test":
                # Single prediction - gauge
                prob = res['proba'][0]
                pred = res['pred'][0]
                true = res['true'][0]
                correct = (pred == true)
                
                fig = ChartComponents.create_gauge_chart(prob, threshold)
                st.plotly_chart(fig, use_container_width=True)
                
                # Result card
                if pred == 1 and correct:
                    st.error(f"**SALDIRI TESPİT EDİLDİ!**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ✓ Doğru")
                elif pred == 1 and not correct:
                    st.warning(f"**YANLIŞLIKLA ALARM!**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ✗ False Positive")
                elif pred == 0 and correct:
                    st.success(f"**NORMAL TRAFİK**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ✓ Doğru")
                else:
                    st.error(f"**SALDIRI KAÇIRILDI!**\n\nOlasılık: {prob*100:.1f}% | Sonuç: ✗ False Negative")
            
            else:
                # Batch/Simulation - summary stats
                metrics = MetricsService.calculate_all_metrics(res['true'], res['pred'], res['proba'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Doğruluk", f"{metrics['accuracy']*100:.1f}%")
                with col2:
                    st.metric("True Pos", metrics['tp'])
                with col3:
                    st.metric("False Pos", metrics['fp'])
                with col4:
                    st.metric("False Neg", metrics['fn'])
                
                # Distribution chart
                fig = ChartComponents.create_probability_distribution(
                    res['proba'][res['true'] == 0],
                    res['proba'][res['true'] == 1],
                    threshold
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Soldaki panelden 'TAHMİN YAP!' butonuna basın!")


def render_realtime_monitoring_tab(
    model_service: ModelService,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float
) -> None:
    """Real-time monitoring tab'ını render et"""
    
    monitoring_service = MonitoringService()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Yeni Monitoring Başlat", key="monitoring", type="primary"):
            # Generate time series
            data = monitoring_service.generate_traffic_data(
                n_points=MONITORING_POINTS,
                attack_probability=0.15
            )
            st.session_state['monitoring_data'] = data
            st.session_state['monitoring_start_time'] = datetime.now()
    
    with col2:
        auto_refresh = st.checkbox("Auto-Refresh", value=False, key="auto_refresh")
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh (sn):",
                REFRESH_INTERVAL_MIN,
                REFRESH_INTERVAL_MAX,
                DEFAULT_REFRESH_INTERVAL,
                key="refresh_interval"
            )
            st.info(f"Her {refresh_interval} saniyede yenilenecek")
            
            # Auto-refresh için Streamlit'in rerun özelliğini kullan
            if 'last_refresh' not in st.session_state:
                st.session_state['last_refresh'] = datetime.now()
            
            elapsed = (datetime.now() - st.session_state['last_refresh']).total_seconds()
            if elapsed >= refresh_interval:
                # Yeni veri noktası ekle
                if 'monitoring_data' in st.session_state:
                    data = st.session_state['monitoring_data']
                    last_timestamp = data['timestamps'][-1]
                    new_point = monitoring_service.generate_live_update(last_timestamp)
                    
                    # Veriyi güncelle (son 100 noktayı tut)
                    data['timestamps'].append(new_point['timestamp'])
                    data['total'] = np.append(data['total'], new_point['total'])
                    data['normal'] = np.append(data['normal'], new_point['normal'])
                    data['attacks'] = np.append(data['attacks'], new_point['attack'])
                    data['attack_traffic'] = np.append(data['attack_traffic'], new_point['attack_traffic'])
                    
                    # Son 100 noktayı tut
                    if len(data['timestamps']) > MONITORING_POINTS:
                        data['timestamps'] = data['timestamps'][-MONITORING_POINTS:]
                        data['total'] = data['total'][-MONITORING_POINTS:]
                        data['normal'] = data['normal'][-MONITORING_POINTS:]
                        data['attacks'] = data['attacks'][-MONITORING_POINTS:]
                        data['attack_traffic'] = data['attack_traffic'][-MONITORING_POINTS:]
                    
                    st.session_state['monitoring_data'] = data
                    st.session_state['last_refresh'] = datetime.now()
                    st.rerun()
    
    if 'monitoring_data' in st.session_state:
        data = st.session_state['monitoring_data']
        metrics = monitoring_service.calculate_metrics(data)
        
        # Real-time metrics dashboard
        met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
        
        with met_col1:
            delta = np.random.randint(-500, 1000)
            st.metric(
                "Toplam Trafik",
                f"{int(metrics['total_traffic']):,}",
                delta=f"{delta:+d} pkt"
            )
        with met_col2:
            delta_attack = np.random.choice(['+', '-']) + str(np.random.randint(1, 5))
            st.metric(
                "Saldırı Sayısı",
                metrics['attack_count'],
                delta=delta_attack
            )
        with met_col3:
            delta_rate = np.random.choice(['+', '-']) + f"{np.random.uniform(0.5, 2):.1f}%"
            st.metric(
                "Saldırı Oranı",
                f"{metrics['attack_rate']:.1f}%",
                delta=delta_rate
            )
        with met_col4:
            delta_latency = f"-{np.random.uniform(0.1, 0.5):.2f}ms"
            st.metric(
                "Avg Latency",
                f"{metrics['avg_latency']:.2f}ms",
                delta=delta_latency
            )
        with met_col5:
            delta_detection = f"+{np.random.uniform(0.1, 0.8):.1f}%"
            st.metric(
                "Detection Rate",
                f"{metrics['detection_rate']:.1f}%",
                delta=delta_detection
            )
        
        st.markdown("---")
        
        # Traffic chart
        fig = ChartComponents.create_traffic_monitoring_chart(
            data['timestamps'],
            data['total'],
            data['attacks']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert Panel
        if metrics['attack_count'] > 0:
            st.markdown('<div style="display:flex;align-items:center;gap:8px;font-size:1.3rem;font-weight:600;margin-top:1rem;"><i data-lucide="alert-triangle" style="width:24px;height:24px;color:#ef4444;"></i><span>Recent Alerts</span></div>', unsafe_allow_html=True)
            
            alert_col1, alert_col2 = st.columns([3, 1])
            
            with alert_col1:
                # Son 20 noktadaki saldırıları göster
                recent_attacks = [
                    (data['timestamps'][i], data['attacks'][i])
                    for i in range(max(0, len(data['timestamps'])-20), len(data['timestamps']))
                    if data['attacks'][i] == 1
                ]
                
                for idx, (t, a) in enumerate(recent_attacks[:3]):  # En fazla 3 alert
                    st.error(
                        f"**ALERT #{idx+1}** | "
                        f"Time: {t.strftime('%H:%M:%S')} | "
                        f"Threat: High | "
                        f"Action: Blocked"
                    )
            
            with alert_col2:
                st.markdown("**Threat Level**")
                threat_level, threat_percentage = monitoring_service.get_threat_level(
                    metrics['attack_count'],
                    len(data['attacks'])
                )
                
                if threat_level == "Low":
                    st.success(f"{threat_level}\n\n{threat_percentage}%")
                elif threat_level == "Medium":
                    st.warning(f"{threat_level}\n\n{threat_percentage}%")
                else:
                    st.error(f"{threat_level}\n\n{threat_percentage}%")
    else:
        st.info("'Yeni Monitoring Başlat' butonuna basın!")


def main():
    """Ana fonksiyon"""
    try:
        # Initialize services
        model_service = ModelService()
        models = model_service.load_all_models()
        
        if not models:
            st.error("Modeller bulunamadı. Lütfen modelleri eğitin: `make train_ensemble`")
            return
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Render header
        render_header(models, X_test)
        
        # Render sidebar and get selected model/threshold
        model_name, threshold = render_sidebar(models, y_test, list(models.keys())[0])
        
        # Render performance metrics
        render_performance_metrics(model_service, model_name, X_test, y_test, threshold)
        
        # Main Tabs (Streamlit tabs don't support HTML/icons in labels)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Canlı Demo",
                "Real-Time Monitoring",
                "Model Karşılaştırma",
                "Saldırı Analizi",
                "Performans Detayları",
                "Feature Importance"
            ])
        
        # TAB 1: Live Demo
        with tab1:
            render_live_demo_tab(model_service, model_name, X_test, y_test, threshold)
        
        # TAB 2: Real-Time Monitoring
        with tab2:
            render_realtime_monitoring_tab(model_service, model_name, X_test, y_test, threshold)
        
        # TAB 3: Model Comparison
        with tab3:
            st.info("Implementation in progress")
        
        # TAB 4: Attack Analysis
        with tab4:
            st.info("Implementation in progress")
        
        # TAB 5: Performance Details
        with tab5:
            st.info("Implementation in progress")
        
        # TAB 6: Feature Importance
        with tab6:
            st.info("Implementation in progress")
    
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        st.info("Modelleri eğitin: `make train_ensemble`")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

