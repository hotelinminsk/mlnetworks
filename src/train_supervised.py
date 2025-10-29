import pandas as pd
from sklearn.linear_model import SGDClassifier
from joblib import dump
from pathlib import Path
from src.config import DATA_PROCESSED, MODELS, RANDOM_STATE


def main():
    Path(MODELS).mkdir(parents=True, exist_ok=True)

    # Load processed features and labels
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].values.ravel()

    print(f"Training supervised SGDClassifier on {len(X_train):,} samples...")

    # Stronger regularization to reduce over-reliance on single OHE categories (e.g., service='-')
    clf = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        penalty="elasticnet",
        l1_ratio=0.25,
        alpha=0.0005,
        random_state=RANDOM_STATE,
        max_iter=2000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    dump(clf, MODELS / "supervised_sgd.joblib")
    print("Supervised model trained and saved to models/supervised_sgd.joblib")


if __name__ == "__main__":
    main()
