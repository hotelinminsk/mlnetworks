"""
Tüm modelleri sırayla eğit
SOLID: Single Responsibility - Model eğitimi koordinasyonu
"""
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_ensemble import main as train_ensemble
from src.train_lightgbm import main as train_lightgbm
from src.train_xgboost import main as train_xgboost


def main():
    """Tüm modelleri sırayla eğit"""
    print("="*80)
    print("🚀 TÜM MODELLERİ EĞİTME BAŞLIYOR")
    print("="*80)
    
    models_to_train = [
        ("Ensemble Models (RF, GB, ET)", train_ensemble),
        ("LightGBM", train_lightgbm),
        ("XGBoost", train_xgboost),
    ]
    
    for name, train_func in models_to_train:
        print(f"\n{'='*80}")
        print(f"📊 {name} Eğitiliyor...")
        print(f"{'='*80}\n")
        
        try:
            train_func()
            print(f"✅ {name} başarıyla eğitildi!\n")
        except Exception as e:
            print(f"❌ {name} eğitiminde hata: {e}\n")
            continue
    
    print("="*80)
    print("🎉 TÜM MODELLER EĞİTİMİ TAMAMLANDI!")
    print("="*80)


if __name__ == "__main__":
    main()

