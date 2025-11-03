"""
Comprehensive model comparison script.
Evaluates all trained models and generates comparison reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)

from src.config import DATA_PROCESSED, MODELS

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_test_data():
    """Load test data"""
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()
    return X_test, y_test


def get_available_models():
    """Scan models directory and return available models"""
    model_files = {
        'Isolation Forest': MODELS / 'isolation_forest.joblib',
        'SGD Classifier': MODELS / 'supervised_sgd.joblib',
        'Random Forest': MODELS / 'random_forest.joblib',
        'Gradient Boosting': MODELS / 'gradient_boosting.joblib',
        'Extra Trees': MODELS / 'extra_trees.joblib',
    }

    available = {}
    for name, path in model_files.items():
        if path.exists():
            available[name] = path

    return available


def evaluate_model(model, model_name, X_test, y_test):
    """Comprehensive evaluation of a single model"""

    # Get predictions
    if model_name == 'Isolation Forest':
        scores = -model.decision_function(X_test)  # Higher = more anomalous
    else:
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test)
        else:
            scores = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold using F1
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Predictions with optimal threshold
    y_pred = (scores >= optimal_threshold).astype(int)

    # Calculate comprehensive metrics
    metrics = {
        'ROC AUC': roc_auc_score(y_test, scores),
        'Average Precision': average_precision_score(y_test, scores),
        'Precision': precision[optimal_idx],
        'Recall': recall[optimal_idx],
        'F1 Score': f1_scores[optimal_idx],
        'Accuracy': (y_pred == y_test).mean(),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_test, y_pred),
        'Optimal Threshold': optimal_threshold,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['True Positives'] = tp
        metrics['True Negatives'] = tn
        metrics['False Positives'] = fp
        metrics['False Negatives'] = fn
        metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return metrics, scores, y_pred, optimal_threshold


def plot_roc_comparison(results, y_test, save_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data['scores'])
        auc = data['metrics']['ROC AUC']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC comparison saved to {save_path}")


def plot_pr_comparison(results, y_test, save_path):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))

    for model_name, data in results.items():
        precision, recall, _ = precision_recall_curve(y_test, data['scores'])
        ap = data['metrics']['Average Precision']
        plt.plot(recall, precision, label=f'{model_name} (AP={ap:.4f})', linewidth=2)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR comparison saved to {save_path}")


def plot_metric_comparison(metrics_df, save_path):
    """Plot bar chart comparing key metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    key_metrics = ['ROC AUC', 'F1 Score', 'Precision', 'Recall']

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]
        data = metrics_df[metric].sort_values(ascending=False)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
        bars = ax.barh(data.index, data.values, color=colors)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metric comparison saved to {save_path}")


def plot_confusion_matrices(results, y_test, save_path):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (model_name, data) in enumerate(results.items()):
        cm = confusion_matrix(y_test, data['predictions'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    cbar=True, square=True)

        axes[idx].set_title(f'{model_name}\nAccuracy: {data["metrics"]["Accuracy"]:.4f}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def generate_comparison_report(metrics_df, save_path):
    """Generate markdown report"""
    report = "# Model Comparison Report\n\n"
    report += f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## Executive Summary\n\n"
    best_model = metrics_df['ROC AUC'].idxmax()
    best_auc = metrics_df.loc[best_model, 'ROC AUC']
    report += f"**Best Performing Model:** {best_model}\n\n"
    report += f"**ROC AUC:** {best_auc:.4f}\n\n"

    report += "## Performance Metrics\n\n"
    report += metrics_df[['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy']].to_markdown()
    report += "\n\n"

    report += "## Detailed Metrics\n\n"
    report += metrics_df.to_markdown()
    report += "\n\n"

    report += "## Model Rankings\n\n"
    for metric in ['ROC AUC', 'F1 Score', 'Precision', 'Recall']:
        report += f"### By {metric}\n\n"
        ranking = metrics_df[metric].sort_values(ascending=False)
        for rank, (model, score) in enumerate(ranking.items(), 1):
            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'{rank}.')
            report += f"{medal} **{model}**: {score:.4f}\n\n"
        report += "\n"

    report += "## Confusion Matrix Analysis\n\n"
    for model in metrics_df.index:
        if 'True Positives' in metrics_df.columns:
            tp = metrics_df.loc[model, 'True Positives']
            tn = metrics_df.loc[model, 'True Negatives']
            fp = metrics_df.loc[model, 'False Positives']
            fn = metrics_df.loc[model, 'False Negatives']
            report += f"### {model}\n\n"
            report += f"- True Positives: {tp:,.0f}\n"
            report += f"- True Negatives: {tn:,.0f}\n"
            report += f"- False Positives: {fp:,.0f}\n"
            report += f"- False Negatives: {fn:,.0f}\n"
            report += f"- FPR: {metrics_df.loc[model, 'FPR']:.4f}\n"
            report += f"- TPR: {metrics_df.loc[model, 'TPR']:.4f}\n\n"

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {save_path}")


def main():
    print("="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("="*60)

    # Create output directory
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data()
    print(f"Test samples: {len(y_test):,}")

    # Get available models
    available_models = get_available_models()
    print(f"\nFound {len(available_models)} models:")
    for name in available_models.keys():
        print(f"  - {name}")

    # Evaluate all models
    print("\nEvaluating models...")
    results = {}

    for model_name, model_path in available_models.items():
        print(f"\nEvaluating {model_name}...")
        model = load(model_path)
        metrics, scores, predictions, threshold = evaluate_model(
            model, model_name, X_test, y_test
        )
        results[model_name] = {
            'metrics': metrics,
            'scores': scores,
            'predictions': predictions,
            'threshold': threshold
        }
        print(f"  ROC AUC: {metrics['ROC AUC']:.4f}")
        print(f"  F1 Score: {metrics['F1 Score']:.4f}")

    # Create metrics dataframe
    metrics_df = pd.DataFrame({name: data['metrics'] for name, data in results.items()}).T

    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey Metrics:")
    print(metrics_df[['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy']].to_string())

    # Save metrics
    metrics_df.to_csv(output_dir / "model_comparison_metrics.csv")
    print(f"\nMetrics saved to {output_dir / 'model_comparison_metrics.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_roc_comparison(results, y_test, output_dir / "roc_comparison.png")
    plot_pr_comparison(results, y_test, output_dir / "pr_comparison.png")
    plot_metric_comparison(metrics_df, output_dir / "metric_comparison.png")
    plot_confusion_matrices(results, y_test, output_dir / "confusion_matrices.png")

    # Generate report
    print("\nGenerating report...")
    generate_comparison_report(metrics_df, output_dir / "model_comparison_report.md")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
