from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
DATA_RAW = BASE/"data/raw"
DATA_INTERIM = BASE/"data/interim"
DATA_PROCESSED = BASE/"data/processed"
MODELS = BASE/"models"

RANDOM_STATE = 42
CONTAMINATION = 0.05
SCORE_THRESHOLD = None

SIMULATED_SOURCE_POOL = [
    "192.168.1.10",
    "192.168.1.24",
    "192.168.1.37",
    "192.168.1.58",
    "192.168.2.17",
    "192.168.2.33",
    "10.1.10.42",
    "10.1.10.84",
]

SIMULATED_TARGET_POOL = [
    "10.0.0.10",
    "10.0.0.15",
    "10.0.0.21",
    "10.0.1.5",
]
