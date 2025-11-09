# Model Eğitimi Rehberi

## 📊 Mevcut Modeller

### 1. **Isolation Forest** ✅
- **Tip**: Anomaly Detection
- **Dosya**: `isolation_forest.joblib`
- **Eğitim**: `python -m src.train_iforest`
- **Durum**: Eğitildi (n_jobs=1)

### 2. **SGD Classifier** ✅
- **Tip**: Linear Classifier
- **Dosya**: `supervised_sgd.joblib`
- **Eğitim**: `python -m src.train_supervised`
- **Durum**: Eğitildi

### 3. **Random Forest** ⚠️
- **Tip**: Ensemble (Tree-based)
- **Dosya**: `random_forest.joblib`
- **Eğitim**: `python -m src.train_ensemble`
- **Durum**: Eğitilmeli
- **Beklenen ROC AUC**: ~0.9845

### 4. **Gradient Boosting** ⚠️
- **Tip**: Ensemble (Gradient Boosting)
- **Dosya**: `gradient_boosting.joblib`
- **Eğitim**: `python -m src.train_ensemble`
- **Durum**: Eğitilmeli
- **Beklenen ROC AUC**: ~0.9860 (EN İYİ)

### 5. **Extra Trees** ⚠️
- **Tip**: Ensemble (Extremely Randomized Trees)
- **Dosya**: `extra_trees.joblib`
- **Eğitim**: `python -m src.train_ensemble`
- **Durum**: Eğitilmeli
- **Beklenen ROC AUC**: ~0.9848

### 6. **LightGBM** ⚠️
- **Tip**: Gradient Boosting (Microsoft)
- **Dosya**: `lightgbm.joblib`
- **Eğitim**: `python -m src.train_lightgbm`
- **Durum**: Eğitilmeli
- **Özellik**: Hızlı eğitim, yüksek performans

### 7. **XGBoost** ⚠️
- **Tip**: Gradient Boosting (Extreme)
- **Dosya**: `xgboost.joblib`
- **Eğitim**: `python -m src.train_xgboost`
- **Durum**: Eğitilmeli
- **Özellik**: Yüksek performans, early stopping

## 🚀 Hızlı Başlangıç

### Tüm Modelleri Eğit (Önerilen)
```bash
cd mlnetworks
./venv/bin/python -m src.train_all_models
```

### Tek Tek Eğit
```bash
# Ensemble Models (RF, GB, ET)
./venv/bin/python -m src.train_ensemble

# LightGBM
./venv/bin/python -m src.train_lightgbm

# XGBoost
./venv/bin/python -m src.train_xgboost
```

## 📈 Beklenen Performans

| Model | ROC AUC | Eğitim Süresi | Notlar |
|-------|---------|---------------|--------|
| Gradient Boosting | **0.9860** | Orta | ⭐ EN İYİ |
| Extra Trees | 0.9848 | Hızlı | - |
| Random Forest | 0.9845 | Orta | - |
| LightGBM | ~0.985+ | Hızlı | Early stopping |
| XGBoost | ~0.985+ | Orta | Early stopping |
| SGD Classifier | ~0.95 | Çok Hızlı | Linear |
| Isolation Forest | ~0.90 | Orta | Anomaly detection |

## ⚙️ Yapılandırma

Tüm modeller `app/config.py` dosyasında tanımlı:

```python
MODEL_CONFIGS = {
    "Isolation Forest": {...},
    "SGD Classifier": {...},
    "Random Forest": {...},
    "Gradient Boosting": {...},
    "Extra Trees": {...},
    "LightGBM": {...},      # ✨ YENİ
    "XGBoost": {...},       # ✨ YENİ
}
```

## 🔧 Düzeltmeler

- ✅ `n_jobs=-1` → `n_jobs=1` (parallelization sorunu)
- ✅ `verbose=1` → `verbose=0` (daha temiz output)
- ✅ Isolation Forest düzeltildi

## 📝 Notlar

- Tüm modeller `n_jobs=1` ile eğitiliyor (parallelization sorununu önlemek için)
- Model dosyaları `models/` klasörüne kaydediliyor
- Feature importance CSV'leri de kaydediliyor
- Early stopping LightGBM ve XGBoost'ta aktif

