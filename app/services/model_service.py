"""
Model Service Layer
SOLID: Single Responsibility - Model yükleme ve tahmin işlemleri
Dependency Inversion: Interface kullanarak bağımlılıkları azaltıyoruz
"""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from joblib import load

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import MODELS_DIR, MODEL_CONFIGS


class ModelService:
    """Model yükleme ve tahmin işlemleri için service sınıfı"""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self._models: Dict[str, Any] = {}
    
    def load_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Tüm modelleri yükle
        
        Returns:
            Dict: Model adı -> model bilgileri
        """
        models = {}
        
        for name, config in MODEL_CONFIGS.items():
            model_path = self.models_dir / config['file']
            if model_path.exists():
                try:
                    model = load(model_path)
                    models[name] = {
                        'model': model,
                        'icon': config['icon'],
                        'type': config['type']
                    }
                except Exception as e:
                    print(f"Warning: Could not load {name}: {e}")
        
        self._models = models
        return models
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Model ile olasılık tahmini yap
        
        Args:
            model_name: Model adı
            X: Özellik matrisi
            
        Returns:
            np.ndarray: Saldırı olasılıkları
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]['model']
        
        # Isolation Forest için özel işlem
        if model_name == "Isolation Forest":
            scores = -model.decision_function(X)
            # Normalize to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores
        else:
            # Diğer modeller için predict_proba
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            else:
                raise ValueError(f"Model {model_name} does not support predict_proba")
    
    def predict(self, model_name: str, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Model ile binary tahmin yap
        
        Args:
            model_name: Model adı
            X: Özellik matrisi
            threshold: Karar eşiği
            
        Returns:
            np.ndarray: Binary tahminler (0/1)
        """
        proba = self.predict_proba(model_name, X)
        return (proba >= threshold).astype(int)
    
    def get_model(self, model_name: str) -> Any:
        """Model objesini döndür"""
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        return self._models[model_name]['model']
    
    def has_feature_importance(self, model_name: str) -> bool:
        """Model feature importance desteği var mı kontrol et"""
        if model_name not in self._models:
            return False
        
        model = self._models[model_name]['model']
        return hasattr(model, 'feature_importances_')
    
    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """Feature importance değerlerini döndür"""
        if not self.has_feature_importance(model_name):
            return None
        
        model = self._models[model_name]['model']
        return model.feature_importances_

