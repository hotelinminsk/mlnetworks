"""
Dashboard Configuration Constants
Clean Code: Magic numbers ve hard-coded değerler burada toplanır
"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

# Model Files Configuration
MODEL_CONFIGS = {
    "Isolation Forest": {
        "file": "isolation_forest.joblib",
        "icon": "tree-deciduous",
        "type": "anomaly"
    },
    "SGD Classifier": {
        "file": "supervised_sgd.joblib",
        "icon": "zap",
        "type": "linear"
    },
    "Random Forest": {
        "file": "random_forest.joblib",
        "icon": "trees",
        "type": "ensemble"
    },
    "Gradient Boosting": {
        "file": "gradient_boosting.joblib",
        "icon": "rocket",
        "type": "ensemble"
    },
    "Extra Trees": {
        "file": "extra_trees.joblib",
        "icon": "tree-palm",
        "type": "ensemble"
    },
    "LightGBM": {
        "file": "lightgbm.joblib",
        "icon": "gauge",
        "type": "gradient_boosting"
    },
    "XGBoost": {
        "file": "xgboost.joblib",
        "icon": "trending-up",
        "type": "gradient_boosting"
    },
}

# Default Values
DEFAULT_THRESHOLD = 0.7
DEFAULT_MODEL_INDEX = 3  # Gradient Boosting
DEFAULT_SAMPLES = 10
DEFAULT_TOP_N_FEATURES = 20

# UI Configuration
PAGE_CONFIG = {
    "page_title": "Network IDS",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Threshold Configuration
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
THRESHOLD_STEP = 0.05

# Sample Configuration
MIN_SAMPLES = 5
MAX_SAMPLES = 20
DEFAULT_BATCH_SAMPLES = 10
MIN_SIMULATION = 10
MAX_SIMULATION = 100
DEFAULT_SIMULATION = 50

# Feature Importance Configuration
MIN_FEATURES = 10
MAX_FEATURES = 50

# Monitoring Configuration
MONITORING_POINTS = 100
REFRESH_INTERVAL_MIN = 1
REFRESH_INTERVAL_MAX = 10
DEFAULT_REFRESH_INTERVAL = 5

# Traffic Threshold
TRAFFIC_THRESHOLD = 200

# Colors
COLOR_NORMAL = "green"
COLOR_ATTACK = "red"
COLOR_GRADIENT_1 = "#667eea"
COLOR_GRADIENT_2 = "#764ba2"

MODERN_COLORS = {
    # Premium Gradients
    'primary': '#6366f1',        # Indigo-500
    'primary_dark': '#4f46e5',   # Indigo-600
    'secondary': '#8b5cf6',      # Violet-500
    'secondary_dark': '#7c3aed', # Violet-600
    
    # Semantic Colors
    'success': '#10b981',        # Emerald-500
    'success_light': '#34d399',  # Emerald-400
    'danger': '#ef4444',         # Red-500
    'danger_light': '#f87171',   # Red-400
    'warning': '#f59e0b',        # Amber-500
    'warning_light': '#fbbf24',  # Amber-400
    'info': '#3b82f6',           # Blue-500
    'info_light': '#60a5fa',     # Blue-400
    
    # Neutrals
    'light': '#f8fafc',          # Slate-50
    'dark': '#0f172a',           # Slate-900
    'gray': '#64748b',           # Slate-500
    'gray_light': '#94a3b8',     # Slate-400
    'gray_dark': '#334155',      # Slate-700
    
    # Specific Use Cases
    'gradient_start': '#6366f1',
    'gradient_end': '#8b5cf6',
    'normal_traffic': '#10b981',
    'attack_traffic': '#ef4444',
    'background': '#f8fafc',
    'grid': '#e5e7eb',           # Gray-200
    'plot_bg': 'rgba(248, 250, 252, 0.8)',
    'title_color': '#0f172a',
    'label_color': '#64748b',
    
    # Chart Specific
    'chart_bg': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'glass_bg': 'rgba(255, 255, 255, 0.1)',
    'shadow': 'rgba(99, 102, 241, 0.1)'
}

# Animation settings

