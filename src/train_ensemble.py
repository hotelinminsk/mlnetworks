import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from joblib import dump
from pathlib import Path
from src.config import DATA_PROCESSED, MODELS, RANDOM_STATE


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("\n" + "="*60)
    print("Training Random Forest (Optuna-optimized)")
    print("="*60)

    # Optimized hyperparameters from Optuna (ROC AUC: 0.9845)
    clf = RandomForestClassifier(
        n_estimators=168,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=1,  # Parallelization sorununu önlemek için
        verbose=0
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Save model
    dump(clf, MODELS / "random_forest.joblib")
    print(f"Model saved to {MODELS / 'random_forest.joblib'}")

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv(MODELS / "random_forest_feature_importance.csv", index=False)

    return clf, auc


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting model"""
    print("\n" + "="*60)
    print("Training Gradient Boosting (Optuna-optimized)")
    print("="*60)

    # Optimized hyperparameters from Optuna (ROC AUC: 0.9860) - BEST MODEL
    clf = GradientBoostingClassifier(
        n_estimators=370,
        learning_rate=0.01762491334043672,
        max_depth=12,
        min_samples_split=16,
        min_samples_leaf=3,
        subsample=0.8618286290542155,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        verbose=0
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Save model
    dump(clf, MODELS / "gradient_boosting.joblib")
    print(f"Model saved to {MODELS / 'gradient_boosting.joblib'}")

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv(MODELS / "gradient_boosting_feature_importance.csv", index=False)

    return clf, auc


def train_extra_trees(X_train, y_train, X_test, y_test):
    """Train Extra Trees model"""
    print("\n" + "="*60)
    print("Training Extra Trees (Optuna-optimized)")
    print("="*60)

    # Optimized hyperparameters from Optuna (ROC AUC: 0.9848)
    clf = ExtraTreesClassifier(
        n_estimators=137,
        max_depth=11,
        min_samples_split=14,
        min_samples_leaf=1,
        max_features=None,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=1,  # Parallelization sorununu önlemek için
        verbose=0
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Save model
    dump(clf, MODELS / "extra_trees.joblib")
    print(f"Model saved to {MODELS / 'extra_trees.joblib'}")

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv(MODELS / "extra_trees_feature_importance.csv", index=False)

    return clf, auc


def main():
    Path(MODELS).mkdir(parents=True, exist_ok=True)

    # Load processed features and labels
    print("Loading data...")
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].values.ravel()
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")

    # Train all models
    results = {}

    rf_model, rf_auc = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = rf_auc

    gb_model, gb_auc = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['Gradient Boosting'] = gb_auc

    et_model, et_auc = train_extra_trees(X_train, y_train, X_test, y_test)
    results['Extra Trees'] = et_auc

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:20s}: ROC AUC = {auc:.4f}")

    # Save summary
    pd.DataFrame(results.items(), columns=['Model', 'ROC_AUC']).to_csv(
        MODELS / "ensemble_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
