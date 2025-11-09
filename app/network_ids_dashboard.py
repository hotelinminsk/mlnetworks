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
from typing import Dict, Any

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
from components.metrics_display import MetricsDisplay
from components.chart_components import ChartComponents
from src.config import DATA_PROCESSED

# Page configuration
st.set_page_config(**PAGE_CONFIG)

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


@st.cache_resource
def load_test_data():
    """Test verisini yükle"""
    X = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values
    return X, y


def render_header(models: Dict[str, Any], X_test: pd.DataFrame) -> None:
    """Header bölümünü render et"""
    st.markdown(
        '<p class="main-header">🔒 Network Intrusion Detection System</p>',
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
    st.markdown("### 📊 Sistem Performansı")
    
    # Get model predictions
    y_proba = model_service.predict_proba(model_name, X_test)
    y_pred = model_service.predict(model_name, X_test, threshold)
    
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
                <small style="opacity: 0.7;">✨ {model_name}</small>
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
                <small style="opacity: 0.7;">🎯 Decision Threshold</small>
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
                <small style="opacity: 0.7;">📉 False Positives</small>
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
                <small style="opacity: 0.7;">🎯 Saldırı Bulma</small>
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
        st.markdown("### ⚙️ Sistem Kontrolleri")
        
        model_name = st.selectbox(
            "🎯 Model Seçimi",
            list(models.keys()),
            index=DEFAULT_MODEL_INDEX if DEFAULT_MODEL_INDEX < len(models) else 0,
            help="Farklı algoritmaları test edin"
        )
        
        threshold = st.slider(
            "🎚️ Karar Eşiği",
            THRESHOLD_MIN, THRESHOLD_MAX, DEFAULT_THRESHOLD, THRESHOLD_STEP,
            help="Saldırı tespiti için minimum olasılık"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Hızlı İstatistikler")
        
        MetricsDisplay.display_summary_stats(y_test)
        
        st.markdown("---")
        st.markdown("### 🏆 Model Durumu")
        st.success(f"✅ {len(models)} Model Yüklü")
        st.info(f"🎯 Aktif: {models[model_name]['icon']} {model_name}")
        
        return model_name, threshold


def render_live_demo_tab(
    model_service: ModelService,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float
) -> None:
    """Canlı demo tab'ını render et"""
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
            n_samples = st.slider("Örnek Sayısı:", MIN_SAMPLES, MAX_SAMPLES, DEFAULT_BATCH_SAMPLES)
        else:  # Rastgele simülasyon
            traffic_type = "🎲 Rastgele"
            n_samples = st.slider("Simülasyon Boyutu:", MIN_SIMULATION, MAX_SIMULATION, DEFAULT_SIMULATION)
            
            if st.button("🔄 YENİ SİMÜLASYON", type="primary", use_container_width=True):
                st.session_state.clear()
        
        st.markdown("---")
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
        st.markdown("### 📊 Tahmin Sonuçları")
        
        if 'comparison_results' in st.session_state:
            st.markdown("#### 🆚 Model Karşılaştırması")
            for m_name, res in st.session_state['comparison_results'].items():
                accuracy = np.mean(res['predictions'] == res['true']) * 100
                st.markdown(f"**{model_service._models[m_name]['icon']} {m_name}**")
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
                
                fig = ChartComponents.create_gauge_chart(prob, threshold)
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
            st.info("👈 Soldaki panelden 'TAHMİN YAP!' butonuna basın!")


def main():
    """Ana fonksiyon"""
    try:
        # Initialize services
        model_service = ModelService()
        models = model_service.load_all_models()
        
        if not models:
            st.error("❌ Modeller bulunamadı. Lütfen modelleri eğitin: `make train_ensemble`")
            return
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Render header
        render_header(models, X_test)
        
        # Render sidebar and get selected model/threshold
        model_name, threshold = render_sidebar(models, y_test, list(models.keys())[0])
        
        # Render performance metrics
        render_performance_metrics(model_service, model_name, X_test, y_test, threshold)
        
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
            render_live_demo_tab(model_service, model_name, X_test, y_test, threshold)
        
        # TAB 2-6: Diğer tab'lar için placeholder (kısaltma için sadece demo tab'ı implement edildi)
        with tab2:
            st.info("📊 Real-Time Monitoring - Implementation in progress")
        with tab3:
            st.info("🏆 Model Comparison - Implementation in progress")
        with tab4:
            st.info("🔍 Attack Analysis - Implementation in progress")
        with tab5:
            st.info("📈 Performance Details - Implementation in progress")
        with tab6:
            st.info("🧠 Feature Importance - Implementation in progress")
    
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")
        st.info("Modelleri eğitin: `make train_ensemble`")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

