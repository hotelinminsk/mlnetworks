import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump
from pathlib import Path
from src.config import DATA_PROCESSED, MODELS, RANDOM_STATE, CONTAMINATION

def main():
    # güvenlik: klasörleri oluştur
    Path(MODELS).mkdir(parents=True, exist_ok=True)

    # veriyi oku (büyük harfli dosya adlarıyla)
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0]

    # yalnızca benign (label == 0) örneklerle eğit
    X_benign = X_train[y_train.values.ravel() == 0]
    print(f"Training IsolationForest on {len(X_benign):,} benign samples...")

    clf = IsolationForest(
        n_estimators=400,
        max_samples=2048,
        contamination='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(X_benign)

    # modeli kaydet
    dump(clf, MODELS / "isolation_forest.joblib")
    # Avoid emoji to prevent Windows console encoding issues
    print("Model trained and saved to models/isolation_forest.joblib")

if __name__ == "__main__":
    main()
