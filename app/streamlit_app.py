import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Ensure project root is on sys.path so `src` can be imported when running Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, MODELS, DATA_RAW


st.set_page_config(page_title="Network Intrusion Detection", layout="wide")
st.title("🔒 Network Intrusion Detection Dashboard")
st.markdown("**Ağ trafiği saldırı tespiti için makine öğrenmesi modelleri**")


@st.cache_resource
def load_models():
    models = {}
    if (MODELS / "isolation_forest.joblib").exists():
        models["IsolationForest"] = load(MODELS / "isolation_forest.joblib")
    if (MODELS / "supervised_sgd.joblib").exists():
        models["Supervised (SGD)"] = load(MODELS / "supervised_sgd.joblib")
    return models


@st.cache_resource
def load_processed():
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()
    return X_test, y_test


def compute_scores(model_name, model, X):
    if model_name == "IsolationForest":
        scores = model.decision_function(X)
        pos_scores = -scores  # higher => more likely intrusion
        score_label = "Anomaly Score"
    else:
        # Supervised linear model
        scores = model.decision_function(X)
        pos_scores = scores  # higher => more likely intrusion
        score_label = "Decision Score"
    return pos_scores, score_label


def plot_curves(y, s, title_prefix):
    # ROC
    fpr, tpr, _ = roc_curve(y, s)
    auc_roc = roc_auc_score(y, s)
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"ROC AUC={auc_roc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"{title_prefix} ROC Curve")
    ax1.legend(loc="lower right")

    # PR
    p, r, _ = precision_recall_curve(y, s)
    fig2, ax2 = plt.subplots()
    ax2.plot(r, p)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix} Precision-Recall Curve")
    return fig1, fig2


models = load_models()
if not models:
    st.warning("⚠️ Modeller bulunamadı. Önce modelleri eğitin.")
    st.stop()

X_test, y_test = load_processed()

# Sidebar - Kontroller
with st.sidebar:
    st.header("⚙️ Model Ayarları")
    
    model_name = st.selectbox(
        "Model Seçin",
        list(models.keys()),
        help="İki model seçeneği:\n"
             "• Isolation Forest: Anomali tespiti (sadece normal trafikle eğitildi)\n"
             "• Supervised SGD: Denetimli öğrenme (normal + saldırı örnekleriyle eğitildi)"
    )
    model = models[model_name]

    st.markdown("---")
    st.subheader("📊 Model Bilgileri")
    
    scores, score_label = compute_scores(model_name, model, X_test)
    st.info(f"**Skor Tipi:** {score_label}\n\n"
            f"Yüksek skor = Saldırı olasılığı daha yüksek")

    # Auto threshold by F1
    p, r, thr = precision_recall_curve(y_test, scores)
    eps = 1e-12
    p1, r1 = p[1:], r[1:]
    f1 = 2 * p1 * r1 / (p1 + r1 + eps)
    default_thr = float(thr[int(np.nanargmax(f1))]) if thr.size else 0.0
    
    current_precision = p1[int(np.nanargmax(f1))] if f1.size > 0 else 0
    current_recall = r1[int(np.nanargmax(f1))] if f1.size > 0 else 0

    st.markdown("---")
    st.subheader("🎯 Eşik (Threshold) Ayarı")
    st.caption("Eşik değeri, skorun saldırı olarak kabul edileceği minimum değeri belirler.\n"
               "Daha yüksek eşik = Daha az ama daha emin pozitif tahminleri\n"
               "Daha düşük eşik = Daha çok pozitif tahmin (daha fazla yakalama)")
    
    thr_val = st.slider(
        "Eşik Değeri",
        float(np.min(scores)),
        float(np.max(scores)),
        float(default_thr),
        help=f"Otomatik önerilen eşik: {default_thr:.4f}\n"
             f"(F1 skoruna göre optimize edildi)"
    )
    
    # Show current metrics at this threshold
    pred_temp = (scores >= thr_val).astype(int)
    cm_temp = confusion_matrix(y_test, pred_temp)
    if len(cm_temp) == 2 and cm_temp.size == 4:
        tn, fp, fn, tp = cm_temp.ravel()
        precision_at_thr = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_at_thr = tp / (tp + fn) if (tp + fn) > 0 else 0
        st.metric("Precision (Seçilen Eşikte)", f"{precision_at_thr:.3f}")
        st.metric("Recall (Seçilen Eşikte)", f"{recall_at_thr:.3f}")

    st.markdown("---")
    st.subheader("🔍 Görüntüleme")
    show_raw = st.checkbox("Ham veri tablolarını göster", value=False)


# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["📈 Model Performansı", "🔍 Veri İnceleme", "🚀 Canlı Tahmin"])

with tab1:
    st.header("Model Performans Metrikleri")
    st.markdown("Test seti üzerinde modelin nasıl çalıştığını gösterir.")
    
    pred = (scores >= thr_val).astype(int)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, digits=4, output_dict=True)
    auc = roc_auc_score(y_test, scores)
    
    # Confusion Matrix and Key Metrics
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📊 Karışıklık Matrisi (Confusion Matrix)")
        st.caption("Modelin tahminlerinin gerçek değerlerle karşılaştırılması")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Saldırı"], yticklabels=["Normal", "Saldırı"])
        ax.set_xlabel("Tahmin Edilen")
        ax.set_ylabel("Gerçek")
        ax.set_title(f"{model_name} - Karışıklık Matrisi")
        
        # Add text annotations
        if len(cm) == 2 and cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            ax.text(0.5, -0.15, f"TN: {tn}\nFP: {fp}", ha='center', va='top', 
                   transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.text(1.5, -0.15, f"FN: {fn}\nTP: {tp}", ha='center', va='top',
                   transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📋 Performans Metrikleri")
        
        # Calculate confusion matrix values
        if len(cm) == 2 and cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("✅ Doğru Pozitif (TP)", f"{tp:,}", 
                         help="Saldırı olarak doğru tespit edilen")
                st.metric("❌ Yanlış Pozitif (FP)", f"{fp:,}",
                         help="Normal trafik ama saldırı olarak tahmin edilen")
            with col2b:
                st.metric("❌ Yanlış Negatif (FN)", f"{fn:,}",
                         help="Saldırı ama normal olarak tahmin edilen")
                st.metric("✅ Doğru Negatif (TN)", f"{tn:,}",
                         help="Normal olarak doğru tespit edilen")
        
        st.markdown("---")
        st.markdown("### Ana Metrikler")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Precision (Kesinlik)", f"{report['1']['precision']:.4f}",
                     help="Saldırı olarak tahmin edilenlerin ne kadarının gerçekten saldırı olduğu")
            st.metric("Recall (Hassasiyet)", f"{report['1']['recall']:.4f}",
                     help="Gerçek saldırıların ne kadarının yakalandığı")
        with metric_col2:
            st.metric("F1-Score", f"{report['1']['f1-score']:.4f}",
                     help="Precision ve Recall'un harmonik ortalaması")
            st.metric("Accuracy (Doğruluk)", f"{report['accuracy']:.4f}",
                     help="Toplam doğru tahmin oranı")
        
        st.metric("ROC AUC", f"{auc:.4f}",
                 help="Modelin ayrıştırma yeteneği (1.0 = mükemmel, 0.5 = rastgele)")
        st.caption(f"**Kullanılan Eşik:** {thr_val:.4f}")
    
    st.markdown("---")
    
    # ROC and PR Curves
    col_roc, col_pr = st.columns(2)
    
    with col_roc:
        st.subheader("📈 ROC Eğrisi")
        st.caption("True Positive Rate vs False Positive Rate\n"
                  "AUC ne kadar yüksekse model o kadar iyi")
        fig_roc, _ = plot_curves(y_test, scores, model_name)
        st.pyplot(fig_roc)
    
    with col_pr:
        st.subheader("📉 Precision-Recall Eğrisi")
        st.caption("Precision vs Recall\n"
                  "Dengesiz veri setleri için ROC'tan daha bilgilendiricidir")
        _, fig_pr = plot_curves(y_test, scores, model_name)
        st.pyplot(fig_pr)
    
    # Score Distribution
    st.markdown("---")
    st.subheader("📊 Skor Dağılımları")
    st.caption("Normal ve saldırı trafiğinin skor dağılımları. Kırmızı çizgi eşik değerini gösterir.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.hist(scores[y_test == 0], bins=50, alpha=0.6, label="Normal Trafik", color='green')
    ax3.hist(scores[y_test == 1], bins=50, alpha=0.6, label="Saldırı Trafiği", color='red')
    ax3.axvline(thr_val, color="red", linestyle="--", linewidth=2, label=f"Eşik ({thr_val:.4f})")
    ax3.set_xlabel(score_label)
    ax3.set_ylabel("Örnek Sayısı")
    ax3.set_title(f"{model_name} - Skor Dağılımları")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with tab2:
    st.header("Veri İnceleme ve Analiz")
    st.markdown("Test setindeki örnekleri detaylı inceleyebilir, doğru ve yanlış tahminleri analiz edebilirsiniz.")
    
    df_view = X_test.copy()
    df_view["row"] = np.arange(len(X_test))
    df_view["score"] = scores
    df_view["pred"] = pred
    df_view["label"] = y_test
    df_view["correct"] = (df_view["pred"] == df_view["label"])

    st.markdown("### 🔍 Örnek Filtreleme")
    
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        slice_choice = st.selectbox(
            "Filtre Türü Seçin",
            [
                "Random (Rastgele)",
                "Top False Positives (En yüksek skorlu yanlış pozitifler)",
                "Top False Negatives (En düşük skorlu yanlış negatifler)",
                "True Positives (Doğru tespit edilen saldırılar)",
                "True Negatives (Doğru tespit edilen normal trafik)",
                "False Positives (Normal ama saldırı olarak tahmin edilen)",
                "False Negatives (Saldırı ama normal olarak tahmin edilen)",
            ],
            help="Farklı kategorilerdeki örnekleri incelemek için filtre seçin"
        )
    
    with col_filter2:
        count = st.number_input("Kaç satır gösterilsin?", min_value=1, max_value=100, value=10)
    
    # Map display name to internal name
    slice_map = {
        "Random (Rastgele)": "Random",
        "Top False Positives (En yüksek skorlu yanlış pozitifler)": "Top False Positives",
        "Top False Negatives (En düşük skorlu yanlış negatifler)": "Top False Negatives",
        "True Positives (Doğru tespit edilen saldırılar)": "True Positives",
        "True Negatives (Doğru tespit edilen normal trafik)": "True Negatives",
        "False Positives (Normal ama saldırı olarak tahmin edilen)": "False Positives",
        "False Negatives (Saldırı ama normal olarak tahmin edilen)": "False Negatives",
    }
    slice_internal = slice_map[slice_choice]

    if slice_internal == "Random":
        sample = df_view.sample(n=min(int(count), len(df_view)), random_state=42)
    elif slice_internal == "Top False Positives":
        sample = df_view[(df_view.label == 0) & (df_view.pred == 1)].sort_values("score", ascending=False).head(int(count))
    elif slice_internal == "Top False Negatives":
        sample = df_view[(df_view.label == 1) & (df_view.pred == 0)].sort_values("score", ascending=True).head(int(count))
    elif slice_internal == "True Positives":
        sample = df_view[(df_view.label == 1) & (df_view.pred == 1)].head(int(count))
    elif slice_internal == "True Negatives":
        sample = df_view[(df_view.label == 0) & (df_view.pred == 0)].head(int(count))
    elif slice_internal == "False Positives":
        sample = df_view[(df_view.label == 0) & (df_view.pred == 1)].head(int(count))
    else:  # False Negatives
        sample = df_view[(df_view.label == 1) & (df_view.pred == 0)].head(int(count))
    
    st.markdown("### 📋 Örnek Veriler")
    
    # Show summary stats
    if len(sample) > 0:
        st.info(f"**{len(sample)}** örnek gösteriliyor. "
               f"Ortalama skor: {sample['score'].mean():.4f}, "
               f"Doğru tahmin oranı: {(sample['correct'].sum()/len(sample)*100):.1f}%")
    
    cols_to_show = ["row", "score", "pred", "label", "correct"] + [c for c in df_view.columns if c not in {"row", "score", "pred", "label", "correct"}]
    st.dataframe(sample[cols_to_show], use_container_width=True)
    
    if show_raw:
        st.caption("💡 Tüm özellik sütunları gösteriliyor. 'row' sütununu indeks referansı olarak kullanabilirsiniz.")

@st.cache_resource
def load_preprocessor():
    try:
        ct = load(MODELS / "preprocess_ct.joblib")
    except Exception as e:
        return None, None, None, None
    # Extract column lists and helper estimators
    num_cols = []
    cat_cols = []
    num_pipe = None
    cat_pipe = None
    for name, trans, cols in getattr(ct, "transformers_", []):
        if name == "num":
            num_cols = list(cols)
            num_pipe = ct.named_transformers_["num"]
        elif name == "cat":
            cat_cols = list(cols)
            cat_pipe = ct.named_transformers_["cat"]
    return ct, num_cols, num_pipe, (cat_cols, cat_pipe)


with tab3:
    st.header("🚀 Canlı Tahmin (Live Inference)")
    st.markdown("Yeni ağ trafiği örnekleri girerek modelin anlık tahmin yapmasını sağlayabilirsiniz.")

    ct, num_cols, num_pipe, cat_info = load_preprocessor()
    if ct is None:
        st.error("❌ Ön işleme modeli bulunamadı. Canlı tahmin için models/preprocess_ct.joblib dosyası gerekli.")
    else:
        cat_cols, cat_pipe = cat_info
        num_defaults = {}
        if num_pipe is not None and hasattr(num_pipe.named_steps.get("imputer", None), "statistics_"):
            stats = num_pipe.named_steps["imputer"].statistics_
            for c, val in zip(num_cols, stats):
                num_defaults[c] = float(val) if val is not None else 0.0

        cat_options = {}
        if cat_pipe is not None and hasattr(cat_pipe.named_steps.get("onehot", None), "categories_"):
            cats = cat_pipe.named_steps["onehot"].categories_
            for c, opts in zip(cat_cols, cats):
                # Cast numpy types to native for Streamlit
                cat_options[c] = [None if x is None else str(x) for x in opts]

        # Try to build dynamic presets from raw testing-set; fallback to static
        @st.cache_resource
        def load_dynamic_presets(n_each: int = 5):
            presets = {}
            try:
                df_raw = pd.read_parquet(DATA_RAW / "testing-set.parquet")
                use_cols = [c for c in (num_cols + cat_cols) if c in df_raw.columns]
                # Normal
                df0 = df_raw[df_raw.get("label", 0) == 0]
                for i, (_, r) in enumerate(df0[use_cols].head(n_each).iterrows(), start=1):
                    presets[f"Normal-{i}"] = {k: (str(v) if k in cat_cols else float(v)) for k, v in r.items()}
                # Intrusions
                df1 = df_raw[df_raw.get("label", 1) == 1]
                for i, (_, r) in enumerate(df1[use_cols].head(n_each).iterrows(), start=1):
                    # Attempt to append attack_cat if available for display only
                    d = {k: (str(v) if k in cat_cols else float(v)) for k, v in r.items()}
                    presets[f"Intrusion-{i}"] = d
            except Exception as e:
                pass
            return presets

        DYN_PRESETS = load_dynamic_presets(5)

        # Static fallback presets (subset of columns)
        STATIC_PRESETS = {
        "Normal-1": {
            "label": 0,
            "dur": 0.000011,
            "proto": "udp",
            "service": "-",
            "state": "INT",
            "spkts": 2,
            "dpkts": 0,
            "sbytes": 496,
            "dbytes": 0,
            "rate": 90909.09375,
            "sload": 180363632.0,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 0.0109999999,
            "dinpkt": 0.0,
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 248,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "is_sm_ips_ports": 0,
        },
        "Normal-2": {
            "label": 0,
            "dur": 0.000008,
            "proto": "udp",
            "service": "-",
            "state": "INT",
            "spkts": 2,
            "dpkts": 0,
            "sbytes": 1762,
            "dbytes": 0,
            "rate": 125000.0,
            "sload": 881000000.0,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 0.0080000004,
            "dinpkt": 0.0,
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 881,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "is_sm_ips_ports": 0,
        },
        "Normal-3": {
            "label": 0,
            "dur": 0.000005,
            "proto": "udp",
            "service": "-",
            "state": "INT",
            "spkts": 2,
            "dpkts": 0,
            "sbytes": 1068,
            "dbytes": 0,
            "rate": 200000.0,
            "sload": 854400000.0,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 0.0049999999,
            "dinpkt": 0.0,
            "sjit": 0.0,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 534,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "is_sm_ips_ports": 0,
        },
        # Intrusion samples
        "Intrusion-1": {
            "label": 1,
            "dur": 0.9219869971,
            "proto": "ospf",
            "service": "-",
            "state": "INT",
            "spkts": 20,
            "dpkts": 0,
            "sbytes": 1280,
            "dbytes": 0,
            "rate": 20.6076660156,
            "sload": 10551.125,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 48.5256347656,
            "dinpkt": 0.0,
            "sjit": 52.2538032532,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 64,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "is_sm_ips_ports": 0,
        },
        "Intrusion-2": {
            "label": 1,
            "dur": 0.9219869971,
            "proto": "ospf",
            "service": "-",
            "state": "INT",
            "spkts": 20,
            "dpkts": 0,
            "sbytes": 1280,
            "dbytes": 0,
            "rate": 20.6076660156,
            "sload": 10551.125,
            "dload": 0.0,
            "sloss": 0,
            "dloss": 0,
            "sinpkt": 48.5256347656,
            "dinpkt": 0.0,
            "sjit": 52.2538032532,
            "djit": 0.0,
            "swin": 0,
            "stcpb": 0,
            "dtcpb": 0,
            "dwin": 0,
            "tcprtt": 0.0,
            "synack": 0.0,
            "ackdat": 0.0,
            "smean": 64,
            "dmean": 0,
            "trans_depth": 0,
            "response_body_len": 0,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
            "is_ftp_login": 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": 0,
            "is_sm_ips_ports": 0,
        },
    }

        st.markdown("### 📝 Trafik Örneği Oluştur")
        st.caption("Hazır örneklerden seçebilir veya manuel olarak özellikleri düzenleyebilirsiniz.")
        
        PRESETS = DYN_PRESETS if DYN_PRESETS else STATIC_PRESETS
        preset_name = st.selectbox(
            "Hazır Örnek Seçin (Opsiyonel)",
            ["None"] + list(PRESETS.keys()),
            help="Hazır normal veya saldırı örneklerinden birini seçerek başlayabilirsiniz"
        )
        
        if preset_name != "None":
            with st.expander("Seçilen örnek değerleri"):
                st.json(PRESETS[preset_name])
        
        st.markdown("**Düzenlemek istediğiniz özellikleri seçin:**")
        st.caption("Seçilmeyen özellikler varsayılan değerlerle (sayısal için ortalama, kategorik için ilk kategori) doldurulur.")
        edit_num = st.multiselect(
            "Sayısal Özellikler",
            options=num_cols,
            default=num_cols[: min(5, len(num_cols))],
            help="Düzenlemek istediğiniz sayısal özellikleri seçin"
        )
        edit_cat = st.multiselect(
            "Kategorik Özellikler",
            options=cat_cols,
            default=cat_cols[: min(5, len(cat_cols))],
            help="Düzenlemek istediğiniz kategorik özellikleri seçin"
        )

        st.markdown("---")
        
        if edit_num or edit_cat:
            st.markdown("### ✏️ Özellik Değerlerini Düzenle")
            user_values = {}
            
            if edit_num:
                st.markdown("**Sayısal Özellikler:**")
                cols_num = st.columns(min(3, len(edit_num)))
                
                # Feature descriptions
                feature_descriptions = {
                    "dur": "Duration: Bağlantı süresi (saniye)",
                    "spkts": "Source Packets: Kaynak taraftan gönderilen paket sayısı",
                    "dpkts": "Destination Packets: Hedef taraftan gönderilen paket sayısı",
                    "sbytes": "Source Bytes: Kaynak taraftan gönderilen toplam byte",
                    "dbytes": "Destination Bytes: Hedef taraftan gönderilen toplam byte",
                    "rate": "Rate: Paket gönderim hızı (paket/saniye)",
                    "sload": "Source Load: Kaynak tarafının yük değeri",
                    "dload": "Destination Load: Hedef tarafının yük değeri",
                    "sloss": "Source Loss: Kaynak tarafta kayıp paket sayısı",
                    "dloss": "Destination Loss: Hedef tarafta kayıp paket sayısı",
                    "sinpkt": "Source Inter-Packet Time: Kaynak paketleri arası süre",
                    "dinpkt": "Destination Inter-Packet Time: Hedef paketleri arası süre",
                    "sjit": "Source Jitter: Kaynak tarafında paket gecikme varyasyonu",
                    "djit": "Destination Jitter: Hedef tarafında paket gecikme varyasyonu",
                    "swin": "Source Window: TCP pencere boyutu (kaynak)",
                    "stcpb": "Source TCP Base Sequence Number: TCP sıra numarası (kaynak)",
                    "dtcpb": "Destination TCP Base Sequence Number: TCP sıra numarası (hedef)",
                    "dwin": "Destination Window: TCP pencere boyutu (hedef)",
                    "tcprtt": "TCP Round Trip Time: TCP gidiş-dönüş süresi",
                    "synack": "SYN-ACK Time: SYN-ACK paketi için geçen süre",
                    "ackdat": "ACK Data Time: ACK ile veri arası süre",
                    "smean": "Source Mean: Kaynak tarafı ortalama paket boyutu",
                    "dmean": "Destination Mean: Hedef tarafı ortalama paket boyutu",
                    "trans_depth": "Transaction Depth: HTTP transaction derinliği",
                    "response_body_len": "Response Body Length: HTTP yanıt gövdesi uzunluğu",
                    "ct_src_dport_ltm": "Connection Count Source-Destination Port: Aynı kaynak-hedef port bağlantı sayısı",
                    "ct_dst_sport_ltm": "Connection Count Destination-Source Port: Aynı hedef-kaynak port bağlantı sayısı",
                    "is_ftp_login": "Is FTP Login: FTP girişi var mı (0/1)",
                    "ct_ftp_cmd": "FTP Command Count: FTP komut sayısı",
                    "ct_flw_http_mthd": "HTTP Method Count: HTTP metod sayısı",
                    "is_sm_ips_ports": "Is Same IP Same Port: Aynı IP ve port kullanımı (0/1)",
                }
                
                for idx, c in enumerate(edit_num):
                    with cols_num[idx % len(cols_num)]:
                        preset_val = PRESETS.get(preset_name, {}).get(c) if preset_name != "None" else None
                        dv = float(preset_val) if preset_val is not None else num_defaults.get(c, 0.0)
                        desc = feature_descriptions.get(c, c)
                        user_values[c] = st.number_input(
                            f"{c}", 
                            value=dv, 
                            key=f"num_{c}",
                            help=desc
                        )
            
            if edit_cat:
                st.markdown("**Kategorik Özellikler:**")
                cols_cat = st.columns(min(3, len(edit_cat)))
                
                cat_descriptions = {
                    "proto": "Protocol: Ağ protokolü (tcp, udp, icmp, vb.)",
                    "service": "Service: Kullanılan servis türü (http, dns, ftp, vb. veya '-' bilinmeyen)",
                    "state": "Connection State: Bağlantı durumu (EST, FIN, CON, INT, vb.)",
                }
                
                for idx, c in enumerate(edit_cat):
                    with cols_cat[idx % len(cols_cat)]:
                        opts = cat_options.get(c, [])
                        preset_val = PRESETS.get(preset_name, {}).get(c) if preset_name != "None" else None
                        default_idx = opts.index(str(preset_val)) if preset_val and str(preset_val) in opts else 0
                        desc = cat_descriptions.get(c, c)
                        user_values[c] = st.selectbox(
                            f"{c}", 
                            options=opts if opts else [""], 
                            index=default_idx, 
                            key=f"cat_{c}",
                            help=desc
                        )

        st.markdown("---")
        
        col_samples, col_service = st.columns(2)
        with col_samples:
            n_samples = st.number_input("Kaç örnek tahmin edilsin?", min_value=1, max_value=10, value=1,
                                       help="Aynı örneği birden fazla kez tahmin etmek için")
        with col_service:
            ignore_service_dash = st.checkbox("Service '-' değerini yok say", value=False,
                                            help="Service değeri '-' ise bilinmeyen olarak işaretle")
        
        colp1, colp2 = st.columns(2)
        with colp1:
            do_predict_preset = st.button("🎯 Hazır Örnekle Tahmin Et", type="primary", use_container_width=True)
        with colp2:
            do_predict = st.button("✏️ Manuel Değerlerle Tahmin Et", use_container_width=True)

        if ("do_predict_preset" in locals() and do_predict_preset) or do_predict:
            # Build input row with all expected raw columns
            row = {}
            # Start from defaults
            for c in num_cols:
                row[c] = num_defaults.get(c, 0.0)
            for c in cat_cols:
                row[c] = (cat_options.get(c, [""])[0] if cat_options.get(c) else "")
            # If Predict Preset pressed, overlay only preset values
            if "do_predict_preset" in locals() and do_predict_preset and preset_name != "None":
                for k, v in PRESETS[preset_name].items():
                    if k in row:
                        row[k] = v
            else:
                # Manual: overlay user-edited fields and optional preset
                if preset_name != "None":
                    for k, v in PRESETS[preset_name].items():
                        if k in row:
                            row[k] = v
                for c in edit_num:
                    if c in row:
                        row[c] = user_values.get(c, row[c])
                for c in edit_cat:
                    if c in row:
                        row[c] = user_values.get(c, row[c])

            # Optional: neutralize service '-' by setting to unseen token (OHE will ignore)
            if ignore_service_dash and "service" in row and str(row["service"]).strip() == "-":
                row["service"] = "__UNK__"

            df_in = pd.DataFrame([row] * int(n_samples))
            X_t = ct.transform(df_in)

            # Score with selected model
            if model_name == "IsolationForest":
                live_scores = -models[model_name].decision_function(X_t)
            else:
                live_scores = models[model_name].decision_function(X_t)
            live_pred = (live_scores >= thr_val).astype(int)

            st.markdown("---")
            st.subheader("🎯 Tahmin Sonuçları")
            
            out_live = df_in.copy()
            out_live["score"] = live_scores
            out_live["pred"] = live_pred
            out_live["prediction_label"] = out_live["pred"].map({0: "Normal", 1: "Saldırı"})
            
            # Show summary
            for idx, (score, pred_label) in enumerate(zip(live_scores, live_pred)):
                col_result1, col_result2, col_result3 = st.columns([2, 2, 1])
                with col_result1:
                    st.metric("Skor", f"{score:.4f}")
                with col_result2:
                    prediction_text = "🔴 SALDIRI" if pred_label == 1 else "🟢 NORMAL"
                    st.metric("Tahmin", prediction_text)
                with col_result3:
                    risk_level = "Yüksek" if score >= thr_val else "Düşük"
                    st.metric("Risk", risk_level)
                
                if idx < len(live_scores) - 1:
                    st.markdown("---")
            
            st.markdown("### 📊 Detaylı Sonuçlar")
            st.dataframe(out_live, use_container_width=True)
            
            st.caption(f"💡 **Eşik Değeri:** {thr_val:.4f} | Skor ≥ Eşik ise saldırı olarak tahmin edilir.")
