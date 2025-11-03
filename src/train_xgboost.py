import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump
from pathlib import Path
from src.config import DATA_PROCESSED, MODELS, RANDOM_STATE


def main():
    Path(MODELS).mkdir(parents=True, exist_ok=True)

    # Load processed features and labels
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].values.ravel()
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()

    print(f"Training XGBoost on {len(X_train):,} samples...")
    print(f"Features: {X_train.shape[1]}")

    # XGBoost parameters optimized for intrusion detection
    params = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle imbalance
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist',  # Faster training
    }

    # Train with early stopping
    clf = xgb.XGBClassifier(**params)

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Quick evaluation
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("XGBoost Training Complete!")
    print("="*60)
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"\nROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Save model
    model_path = MODELS / "xgboost.joblib"
    dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv(MODELS / "xgboost_feature_importance.csv", index=False)
    print(f"Feature importance saved to {MODELS / 'xgboost_feature_importance.csv'}")

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
