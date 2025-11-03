"""
Threshold Optimization for Reducing False Positives

Analyzes the precision-recall trade-off and finds optimal thresholds
for different use cases:
- High Precision (fewer false positives)
- Balanced (F1 score)
- High Recall (catch more attacks)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from joblib import load
from pathlib import Path
import seaborn as sns

# Config
from src.config import DATA_PROCESSED, MODELS


def load_data():
    """Load test data"""
    print("Loading test data...")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values.ravel()
    print(f"Test samples: {len(X_test):,}")
    return X_test, y_test


def analyze_threshold_impact(model_path, X_test, y_test, model_name="Model"):
    """Analyze how different thresholds affect FP/FN trade-off"""
    print(f"\n{'='*70}")
    print(f"Analyzing Threshold Impact: {model_name}")
    print(f"{'='*70}")

    # Load model
    model = load(model_path)

    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate precision-recall curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)

    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)

    # Test different thresholds
    threshold_options = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for threshold in threshold_options:
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        fpr_rate = fp / (fp + tn)
        fnr_rate = fn / (fn + tp)

        results.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'TN': tn,
            'FPR': fpr_rate,
            'FNR': fnr_rate,
            'False Alarms/Day': int(fp / 7),  # Assuming 7 days of data
        })

    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "="*70)
    print("THRESHOLD IMPACT ANALYSIS")
    print("="*70)
    print("\nKey: FP = False Positives (Normal traffic flagged as attack)")
    print("     FN = False Negatives (Attacks missed)")
    print("     FPR = False Positive Rate")
    print()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df.to_string(index=False))

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find optimal thresholds for different objectives
    high_precision_idx = results_df['Precision'].idxmax()
    balanced_idx = results_df['F1 Score'].idxmax()
    low_fp_idx = results_df['FP'].idxmin()

    print("\n🎯 Option 1: MINIMAL FALSE POSITIVES (Recommended for production)")
    print(f"   Threshold: {results_df.loc[low_fp_idx, 'Threshold']}")
    print(f"   False Positives: {int(results_df.loc[low_fp_idx, 'FP']):,} ({results_df.loc[low_fp_idx, 'FPR']:.2%} FPR)")
    print(f"   False Negatives: {int(results_df.loc[low_fp_idx, 'FN']):,}")
    print(f"   Precision: {results_df.loc[low_fp_idx, 'Precision']:.4f}")
    print(f"   Recall: {results_df.loc[low_fp_idx, 'Recall']:.4f}")
    print(f"   Estimated False Alarms/Day: ~{int(results_df.loc[low_fp_idx, 'False Alarms/Day'])}")

    print("\n⚖️  Option 2: BALANCED (Best F1 Score)")
    print(f"   Threshold: {results_df.loc[balanced_idx, 'Threshold']}")
    print(f"   False Positives: {int(results_df.loc[balanced_idx, 'FP']):,} ({results_df.loc[balanced_idx, 'FPR']:.2%} FPR)")
    print(f"   False Negatives: {int(results_df.loc[balanced_idx, 'FN']):,}")
    print(f"   Precision: {results_df.loc[balanced_idx, 'Precision']:.4f}")
    print(f"   Recall: {results_df.loc[balanced_idx, 'Recall']:.4f}")
    print(f"   F1 Score: {results_df.loc[balanced_idx, 'F1 Score']:.4f}")

    print("\n🔒 Option 3: MAXIMUM PRECISION")
    print(f"   Threshold: {results_df.loc[high_precision_idx, 'Threshold']}")
    print(f"   False Positives: {int(results_df.loc[high_precision_idx, 'FP']):,} ({results_df.loc[high_precision_idx, 'FPR']:.2%} FPR)")
    print(f"   False Negatives: {int(results_df.loc[high_precision_idx, 'FN']):,}")
    print(f"   Precision: {results_df.loc[high_precision_idx, 'Precision']:.4f}")
    print(f"   Recall: {results_df.loc[high_precision_idx, 'Recall']:.4f}")

    # Create visualization
    create_threshold_plots(results_df, model_name)

    return results_df, y_proba


def create_threshold_plots(results_df, model_name):
    """Create visualization of threshold impact"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Precision vs Recall
    ax1 = axes[0, 0]
    ax1.plot(results_df['Threshold'], results_df['Precision'],
             marker='o', label='Precision', linewidth=2)
    ax1.plot(results_df['Threshold'], results_df['Recall'],
             marker='s', label='Recall', linewidth=2)
    ax1.plot(results_df['Threshold'], results_df['F1 Score'],
             marker='^', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])

    # Plot 2: False Positives vs False Negatives
    ax2 = axes[0, 1]
    ax2.plot(results_df['Threshold'], results_df['FP'],
             marker='o', label='False Positives', linewidth=2, color='red')
    ax2.plot(results_df['Threshold'], results_df['FN'],
             marker='s', label='False Negatives', linewidth=2, color='orange')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('False Positives vs False Negatives', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: FPR vs FNR
    ax3 = axes[1, 0]
    ax3.plot(results_df['Threshold'], results_df['FPR'],
             marker='o', label='False Positive Rate', linewidth=2, color='red')
    ax3.plot(results_df['Threshold'], results_df['FNR'],
             marker='s', label='False Negative Rate', linewidth=2, color='orange')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Rate', fontsize=12)
    ax3.set_title('Error Rates', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Metrics Heatmap
    ax4 = axes[1, 1]
    metrics_for_heatmap = results_df[['Threshold', 'Precision', 'Recall', 'F1 Score']].set_index('Threshold')
    sns.heatmap(metrics_for_heatmap.T, annot=True, fmt='.3f',
                cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'Score'})
    ax4.set_title('Metrics Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path('reports') / f'threshold_analysis_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    plt.close()


def compare_all_models(X_test, y_test):
    """Compare threshold impact across all models"""
    print("\n" + "="*70)
    print("COMPARING ALL MODELS")
    print("="*70)

    models_to_compare = [
        ('Random Forest', MODELS / 'random_forest.joblib'),
        ('Gradient Boosting', MODELS / 'gradient_boosting.joblib'),
        ('Extra Trees', MODELS / 'extra_trees.joblib'),
    ]

    all_results = {}

    for model_name, model_path in models_to_compare:
        if model_path.exists():
            results_df, y_proba = analyze_threshold_impact(
                model_path, X_test, y_test, model_name
            )
            all_results[model_name] = results_df

    # Create comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON AT DIFFERENT THRESHOLDS")
    print("="*70)

    for threshold in [0.5, 0.7, 0.9]:
        print(f"\n📊 At Threshold = {threshold}:")
        print("-" * 70)
        comparison = []
        for model_name, df in all_results.items():
            row = df[df['Threshold'] == threshold].iloc[0]
            comparison.append({
                'Model': model_name,
                'Precision': f"{row['Precision']:.4f}",
                'Recall': f"{row['Recall']:.4f}",
                'F1': f"{row['F1 Score']:.4f}",
                'FP': int(row['FP']),
                'FN': int(row['FN']),
                'FP/Day': int(row['False Alarms/Day'])
            })

        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))


def main():
    """Main execution"""
    print("="*70)
    print("THRESHOLD OPTIMIZATION FOR FALSE POSITIVE REDUCTION")
    print("="*70)

    # Load data
    X_test, y_test = load_data()

    # Analyze all models
    compare_all_models(X_test, y_test)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\n💡 Next Steps:")
    print("1. Review the threshold analysis plots in reports/")
    print("2. Choose a threshold based on your false positive tolerance")
    print("3. Update API to use the chosen threshold")
    print("4. Update dashboard to show predictions with new threshold")
    print("\n📝 To use in production:")
    print("   In your prediction code, replace:")
    print("   >>> y_pred = model.predict(X)")
    print("   With:")
    print("   >>> y_proba = model.predict_proba(X)[:, 1]")
    print("   >>> y_pred = (y_proba >= 0.7).astype(int)  # Use your chosen threshold")


if __name__ == "__main__":
    main()
