import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import load
from pathlib import Path
from src.config import DATA_PROCESSED, MODELS

def main():
    # Verileri oku
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()

    # Modeli yükle
    clf = load(MODELS / "isolation_forest.joblib")

    # Isolation Forest çıktısı: pozitif -> normal, negatif -> anomalous
    scores = clf.decision_function(X_test)
    y_pred = (scores < 0).astype(int)  # 1 = anomaly, 0 = normal

    print("✅ Evaluation Results\n")

    # Confusion Matrix & classification report
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)

    # ROC-AUC (skorların negatifini kullanıyoruz çünkü düşük skor anomali demek)
    auc = roc_auc_score(y_test, -scores)
    print(f"ROC AUC: {auc:.4f}")

    # Skor dosyasını kaydet (dashboard için)
    out = pd.DataFrame({
        "score": scores,
        "pred": y_pred,
        "label": y_test
    })
    out_path = DATA_PROCESSED / "test_scored.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved results → {out_path}")

if __name__ == "__main__":
    main()
