"""
Multi-Model IDS Service
Supports ALL models including Isolation Forest for comprehensive comparison
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS
from joblib import load


class MultiModelService:
    """
    Service that manages multiple ML models for comparison
    """

    MODEL_NAMES = [
        "gradient_boosting",
        # "xgboost",  # Disabled: Requires libomp.dylib (OpenMP runtime)
        # "lightgbm",  # Disabled: Requires libomp.dylib (OpenMP runtime)
        "random_forest",
        "extra_trees",
        "supervised_sgd",
        "isolation_forest"
    ]

    def __init__(self):
        """Initialize all models"""
        self.models = {}
        self.preprocessor = None
        self.load_all_models()

    def load_all_models(self):
        """Load all available models"""
        # Load preprocessor
        preprocessor_path = MODELS / "preprocess_ct.joblib"
        if preprocessor_path.exists():
            self.preprocessor = load(preprocessor_path)

        # Load all models
        for model_name in self.MODEL_NAMES:
            model_path = MODELS / f"{model_name}.joblib"
            if model_path.exists():
                try:
                    self.models[model_name] = load(model_path)
                    print(f"Loaded {model_name}")
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")

        print(f"Loaded {len(self.models)} models")

    def predict_all(self, features: pd.DataFrame, threshold: float = 0.7, is_preprocessed: bool = False) -> Dict:
        """
        Make predictions with all models

        Args:
            features: Input features (DataFrame)
            threshold: Classification threshold
            is_preprocessed: If True, skip preprocessing (features already preprocessed)

        Returns:
            Dictionary with model names as keys and predictions as values
        """
        # Preprocess only if needed
        if is_preprocessed:
            X = features.values if isinstance(features, pd.DataFrame) else features
        else:
            if self.preprocessor is None:
                return {}

            try:
                X = self.preprocessor.transform(features)
            except Exception as e:
                print(f"Preprocessing error: {e}")
                return {}

        results = {}

        for model_name, model in self.models.items():
            try:
                # Get probability or score
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0][1]
                elif hasattr(model, 'decision_function'):
                    score = model.decision_function(X)[0]
                    # Normalize score to probability-like
                    prob = 1 / (1 + np.exp(-score))
                elif hasattr(model, 'score_samples'):
                    # Isolation Forest
                    score = model.score_samples(X)[0]
                    # Convert to probability (lower score = more anomalous)
                    prob = 1 / (1 + np.exp(score))  # Invert so high prob = anomaly
                else:
                    # Fallback: just use predict
                    pred = model.predict(X)[0]
                    prob = float(pred)

                # Apply threshold
                prediction = 1 if prob >= threshold else 0

                results[model_name] = {
                    'prediction': 'attack' if prediction == 1 else 'normal',
                    'probability': float(prob),
                    'confidence': 'high' if prob >= 0.9 or prob <= 0.1 else ('medium' if prob >= 0.7 or prob <= 0.3 else 'low')
                }

            except Exception as e:
                print(f"Prediction error for {model_name}: {e}")
                results[model_name] = {
                    'prediction': 'error',
                    'probability': 0.0,
                    'confidence': 'low'
                }

        return results

    def get_model_agreement(self, predictions: Dict) -> Dict:
        """
        Calculate agreement between models

        Returns:
            Agreement statistics
        """
        attack_votes = sum(1 for p in predictions.values() if p['prediction'] == 'attack')
        total_models = len(predictions)

        agreement_pct = (attack_votes / total_models * 100) if total_models > 0 else 0

        return {
            'attack_votes': attack_votes,
            'normal_votes': total_models - attack_votes,
            'total_models': total_models,
            'agreement_percentage': agreement_pct,
            'consensus': 'attack' if attack_votes > total_models / 2 else 'normal'
        }

    def compare_models(self, features: pd.DataFrame, is_preprocessed: bool = False) -> pd.DataFrame:
        """
        Compare all models on given features

        Args:
            features: Input features
            is_preprocessed: If True, skip preprocessing

        Returns:
            DataFrame with comparison results
        """
        predictions = self.predict_all(features, is_preprocessed=is_preprocessed)

        comparison = []
        for model_name, result in predictions.items():
            comparison.append({
                'Model': model_name.replace('_', ' ').title(),
                'Prediction': result['prediction'],
                'Probability': result['probability'],
                'Confidence': result['confidence']
            })

        return pd.DataFrame(comparison)

    def get_available_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.models.keys())


if __name__ == "__main__":
    service = MultiModelService()
    print(f"\nAvailable models: {service.get_available_models()}")
