from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
DATA_RAW = BASE/"data/raw"
DATA_INTERIM = BASE/"data/interim"
DATA_PROCESSED = BASE/"data/processed"
MODELS = BASE/"models"

RANDOM_STATE = 42
CONTAMINATION = 0.05
SCORE_THRESHOLD = None