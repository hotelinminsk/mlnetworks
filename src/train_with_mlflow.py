"""
Train all models with MLflow tracking.
This script trains models and logs everything to MLflow for experiment tracking.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from joblib import dump
from pathlib import Path

from src.config import DATA_PROCESSED, MODELS, RANDOM_STATE
from src.mlflow_utils import setup_mlflow, log_model_training


def train_with_tracking(model_class, model_name, params, X_train, y_train, X_test, y_test):
    """Train a model and log to MLflow"""

    print(f"\n{'='*60}")
    print(f"Training {model_name} with MLflow tracking")
    print(f"{'='*60}")

    # Train model
    model = model_class(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }

    # Print metrics
    print(f"\nTest Set Performance:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save model
    model_path = MODELS / f"{model_name.lower().replace(' ', '_')}.joblib"
    dump(model, model_path)

    # Log to MLflow
    tags = {
        'model_type': model_class.__name__,
        'dataset': 'UNSW-NB15',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    }

    artifacts = {
        'model_file': str(model_path),
    }

    run_id = log_model_training(
        model=model,
        model_name=model_name.lower().replace(' ', '_'),
        params=params,
        metrics=metrics,
        artifacts=artifacts,
        tags=tags
    )

    print(f"✓ Model saved to {model_path}")
    print(f"✓ Logged to MLflow (run_id: {run_id})")

    return model, metrics, run_id


def main():
    # Setup MLflow
    setup_mlflow(experiment_name="intrusion-detection")

    Path(MODELS).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].values.ravel()
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")

    # Define models and their parameters
    models_config = [
        {
            'class': RandomForestClassifier,
            'name': 'Random Forest',
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': 0
            }
        },
        {
            'class': GradientBoostingClassifier,
            'name': 'Gradient Boosting',
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'subsample': 0.8,
                'max_features': 'sqrt',
                'random_state': RANDOM_STATE,
                'verbose': 0
            }
        },
        {
            'class': ExtraTreesClassifier,
            'name': 'Extra Trees',
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': 0
            }
        }
    ]

    # Train all models
    results = {}

    for config in models_config:
        model, metrics, run_id = train_with_tracking(
            model_class=config['class'],
            model_name=config['name'],
            params=config['params'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        results[config['name']] = {
            'metrics': metrics,
            'run_id': run_id
        }

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    summary_df = pd.DataFrame({
        name: data['metrics']
        for name, data in results.items()
    }).T

    print("\n" + summary_df[['roc_auc', 'accuracy', 'f1_score']].to_string())

    # Best model
    best_model = summary_df['roc_auc'].idxmax()
    best_auc = summary_df.loc[best_model, 'roc_auc']

    print(f"\n🏆 Best Model: {best_model} (ROC AUC: {best_auc:.4f})")
    print("\n✓ All models trained and logged to MLflow!")
    print("\nTo view results in MLflow UI:")
    print("  mlflow ui")
    print("  Visit: http://localhost:5000")


if __name__ == "__main__":
    main()
