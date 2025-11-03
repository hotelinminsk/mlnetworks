"""
Hyperparameter Optimization using Optuna

Optimizes hyperparameters for:
- Random Forest
- Gradient Boosting
- Extra Trees

Uses cross-validation to find the best parameters that maximize ROC AUC.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import json
import os
from datetime import datetime

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    """Load preprocessed training data"""
    print("Loading preprocessed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()

    print(f"Training data shape: {X_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")

    return X_train, y_train


def objective_random_forest(trial, X_train, y_train):
    """Objective function for Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)

    # Use 3-fold CV for speed
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)

    return scores.mean()


def objective_gradient_boosting(trial, X_train, y_train):
    """Objective function for Gradient Boosting"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }

    model = GradientBoostingClassifier(**params)

    # Use 3-fold CV for speed
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)

    return scores.mean()


def objective_extra_trees(trial, X_train, y_train):
    """Objective function for Extra Trees"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = ExtraTreesClassifier(**params)

    # Use 3-fold CV for speed
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)

    return scores.mean()


def optimize_model(model_name, objective_func, X_train, y_train, n_trials=100):
    """
    Run Optuna optimization for a specific model

    Args:
        model_name: Name of the model
        objective_func: Objective function to optimize
        X_train, y_train: Training data
        n_trials: Number of optimization trials

    Returns:
        best_params: Best hyperparameters found
        best_value: Best ROC AUC score
    """
    print(f"\n{'='*60}")
    print(f"Optimizing {model_name}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction='maximize',
        study_name=f"{model_name}_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective_func(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Sequential trials to avoid memory issues
    )

    print(f"\n✅ Optimization complete!")
    print(f"Best ROC AUC: {study.best_value:.6f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    return study.best_params, study.best_value, study


def save_results(results, output_dir='reports'):
    """Save optimization results to JSON and markdown"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(output_dir, f'optuna_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {json_path}")

    # Save markdown report
    md_path = os.path.join(output_dir, f'optuna_report_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write("# Hyperparameter Optimization Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")

        for model_name, result in results.items():
            f.write(f"### {model_name}\n\n")
            f.write(f"**Best ROC AUC:** {result['best_roc_auc']:.6f}\n\n")
            f.write("**Best Parameters:**\n```python\n")
            for param, value in result['best_params'].items():
                if isinstance(value, str):
                    f.write(f"'{param}': '{value}',\n")
                else:
                    f.write(f"'{param}': {value},\n")
            f.write("```\n\n")

    print(f"✅ Report saved to: {md_path}")


def main():
    """Main optimization pipeline"""
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*60)

    # Load data
    X_train, y_train = load_data()

    # Number of trials per model
    n_trials = 100
    print(f"\nRunning {n_trials} trials per model (this will take ~30-60 minutes)")

    results = {}

    # Optimize Random Forest
    print("\n[1/3] Random Forest")
    rf_params, rf_score, rf_study = optimize_model(
        "Random Forest",
        objective_random_forest,
        X_train, y_train,
        n_trials=n_trials
    )
    results['Random Forest'] = {
        'best_params': rf_params,
        'best_roc_auc': float(rf_score)
    }

    # Optimize Gradient Boosting
    print("\n[2/3] Gradient Boosting")
    gb_params, gb_score, gb_study = optimize_model(
        "Gradient Boosting",
        objective_gradient_boosting,
        X_train, y_train,
        n_trials=n_trials
    )
    results['Gradient Boosting'] = {
        'best_params': gb_params,
        'best_roc_auc': float(gb_score)
    }

    # Optimize Extra Trees
    print("\n[3/3] Extra Trees")
    et_params, et_score, et_study = optimize_model(
        "Extra Trees",
        objective_extra_trees,
        X_train, y_train,
        n_trials=n_trials
    )
    results['Extra Trees'] = {
        'best_params': et_params,
        'best_roc_auc': float(et_score)
    }

    # Save results
    save_results(results)

    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print("\n| Model | Best ROC AUC (CV) |")
    print("|-------|-------------------|")
    for model_name, result in results.items():
        print(f"| {model_name:<20} | {result['best_roc_auc']:.6f} |")

    print("\n✅ Optimization complete!")
    print("\nNext steps:")
    print("1. Review the results in reports/")
    print("2. Update src/train_ensemble.py with best parameters")
    print("3. Retrain models: make train_ensemble")
    print("4. Compare results: make compare")


if __name__ == "__main__":
    main()
