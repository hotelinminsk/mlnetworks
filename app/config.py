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
        "icon": "🌲",
        "type": "anomaly"
    },
    "SGD Classifier": {
        "file": "supervised_sgd.joblib",
        "icon": "⚡",
        "type": "linear"
    },
    "Random Forest": {
        "file": "random_forest.joblib",
        "icon": "🌳",
        "type": "ensemble"
    },
    "Gradient Boosting": {
        "file": "gradient_boosting.joblib",
        "icon": "🚀",
        "type": "ensemble"
    },
    "Extra Trees": {
        "file": "extra_trees.joblib",
        "icon": "🌴",
        "type": "ensemble"
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
    "page_icon": "🔒",
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

