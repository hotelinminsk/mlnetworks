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
st.title("Network Intrusion Detection Dashboard")


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
    st.warning("No models found. Train models first.")
    st.stop()

X_test, y_test = load_processed()

with st.sidebar:
    model_name = st.selectbox("Model", list(models.keys()))
    model = models[model_name]

    scores, score_label = compute_scores(model_name, model, X_test)
    st.write(f"Score: {score_label}")

    # Auto threshold by F1
    p, r, thr = precision_recall_curve(y_test, scores)
    eps = 1e-12
    p1, r1 = p[1:], r[1:]
    f1 = 2 * p1 * r1 / (p1 + r1 + eps)
    default_thr = float(thr[int(np.nanargmax(f1))]) if thr.size else 0.0

    thr_val = st.slider(
        "Decision threshold (positive if score >= threshold)",
        float(np.min(scores)),
        float(np.max(scores)),
        float(default_thr),
    )

    show_raw = st.checkbox("Show raw tables", value=False)


pred = (scores >= thr_val).astype(int)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred, digits=4, output_dict=True)
auc = roc_auc_score(y_test, scores)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.subheader("Metrics")
    st.write(
        {
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
            "roc_auc": auc,
            "threshold": thr_val,
        }
    )

fig_roc, fig_pr = plot_curves(y_test, scores, model_name)
st.subheader("ROC / PR Curves")
st.pyplot(fig_roc)
st.pyplot(fig_pr)

st.subheader("Score Distributions")
fig3, ax3 = plt.subplots()
ax3.hist(scores[y_test == 0], bins=50, alpha=0.6, label="normal")
ax3.hist(scores[y_test == 1], bins=50, alpha=0.6, label="intrusion")
ax3.axvline(thr_val, color="red", linestyle="--", label="threshold")
ax3.set_xlabel(score_label)
ax3.legend()
st.pyplot(fig3)

st.subheader("Sample Explorer")
df_view = X_test.copy()
df_view["row"] = np.arange(len(X_test))
df_view["score"] = scores
df_view["pred"] = pred
df_view["label"] = y_test
df_view["correct"] = (df_view["pred"] == df_view["label"])

slice_choice = st.selectbox(
    "Select a slice",
    [
        "Random",
        "Top False Positives",
        "Top False Negatives",
        "True Positives",
        "True Negatives",
        "False Positives",
        "False Negatives",
    ],
)
count = st.number_input("How many rows?", min_value=1, max_value=100, value=10)

if slice_choice == "Random":
    sample = df_view.sample(n=min(int(count), len(df_view)), random_state=42)
elif slice_choice == "Top False Positives":
    sample = df_view[(df_view.label == 0) & (df_view.pred == 1)].sort_values("score", ascending=False).head(int(count))
elif slice_choice == "Top False Negatives":
    sample = df_view[(df_view.label == 1) & (df_view.pred == 0)].sort_values("score", ascending=True).head(int(count))
elif slice_choice == "True Positives":
    sample = df_view[(df_view.label == 1) & (df_view.pred == 1)].head(int(count))
elif slice_choice == "True Negatives":
    sample = df_view[(df_view.label == 0) & (df_view.pred == 0)].head(int(count))
elif slice_choice == "False Positives":
    sample = df_view[(df_view.label == 0) & (df_view.pred == 1)].head(int(count))
else:  # False Negatives
    sample = df_view[(df_view.label == 1) & (df_view.pred == 0)].head(int(count))

cols_to_show = ["row", "score", "pred", "label", "correct"] + [c for c in df_view.columns if c not in {"row", "score", "pred", "label", "correct"}]
st.dataframe(sample[cols_to_show])

if show_raw:
    st.caption("Showing full feature columns in the table above. Use 'row' as index reference.")

# ---------------------- Live Inference ----------------------
st.header("Live Inference")

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

ct, num_cols, num_pipe, cat_info = load_preprocessor()
if ct is None:
    st.info("Preprocessor not found. Live inference requires models/preprocess_ct.joblib")
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

    with st.expander("Create traffic sample(s)", expanded=True):
        PRESETS = DYN_PRESETS if DYN_PRESETS else STATIC_PRESETS
        preset_name = st.selectbox("Preset example", ["None"] + list(PRESETS.keys()))
        if preset_name != "None":
            st.caption("Preset values (info):")
            st.json(PRESETS[preset_name])
        st.write("Select a subset of features to edit; others use defaults (median for numeric, first category for categoricals).")
        edit_num = st.multiselect(
            "Numeric features to edit",
            options=num_cols,
            default=num_cols[: min(5, len(num_cols))],
        )
        edit_cat = st.multiselect(
            "Categorical features to edit",
            options=cat_cols,
            default=cat_cols[: min(5, len(cat_cols))],
        )

        st.markdown("---")
        user_values = {}
        col_left, col_right = st.columns(2)
        with col_left:
            for c in edit_num:
                dv = num_defaults.get(c, 0.0)
                user_values[c] = st.number_input(f"{c}", value=float(dv))
        with col_right:
            for c in edit_cat:
                opts = cat_options.get(c, [])
                default_opt = opts[0] if opts else ""
                user_values[c] = st.selectbox(f"{c}", options=opts if opts else [""], index=0)

        n_samples = st.number_input("How many identical samples?", min_value=1, max_value=10, value=1)
        ignore_service_dash = st.checkbox("Ignore service '-' (treat as unknown)", value=False)
        colp1, colp2 = st.columns(2)
        with colp1:
            do_predict_preset = st.button("Predict Preset")
        with colp2:
            do_predict = st.button("Predict Manual")

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

        st.subheader("Live Predictions")
        out_live = df_in.copy()
        out_live["score"] = live_scores
        out_live["pred"] = live_pred
        st.dataframe(out_live)
