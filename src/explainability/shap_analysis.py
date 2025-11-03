"""
SHAP-based model explainability for intrusion detection models.
Provides global and local explanations for model predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from joblib import load
from pathlib import Path


class ModelExplainer:
    """Wrapper for SHAP-based model explanations"""

    def __init__(self, model, model_name, X_sample=None):
        """
        Initialize explainer with a trained model.

        Args:
            model: Trained scikit-learn model
            model_name: Name of the model (for visualization)
            X_sample: Sample data for background (optional, uses 100 samples if None)
        """
        self.model = model
        self.model_name = model_name
        self.X_sample = X_sample

        # Create SHAP explainer based on model type
        if hasattr(model, 'tree_'):
            # Single tree models
            self.explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'estimators_'):
            # Tree ensemble models (RF, GB, ET)
            self.explainer = shap.TreeExplainer(model)
        else:
            # Linear models or others - use KernelExplainer
            if X_sample is not None:
                background = shap.sample(X_sample, 100)
                self.explainer = shap.KernelExplainer(model.predict_proba, background)
            else:
                self.explainer = None

    def explain_prediction(self, X, feature_names=None, max_display=10):
        """
        Explain a single prediction with waterfall plot.

        Args:
            X: Single sample (1D array or single row DataFrame)
            feature_names: List of feature names
            max_display: Maximum features to display
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide X_sample for non-tree models.")

        # Get SHAP values
        shap_values = self.explainer.shap_values(X.reshape(1, -1) if X.ndim == 1 else X)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Create explanation object
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(X.flatten()))]

        # Get base value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            if len(self.explainer.expected_value) > 1:
                base_val = self.explainer.expected_value[1]
            else:
                base_val = self.explainer.expected_value[0]
        else:
            base_val = self.explainer.expected_value

        explanation = shap.Explanation(
            values=shap_values[0] if shap_values.ndim > 1 else shap_values,
            base_values=base_val,
            data=X.flatten(),
            feature_names=feature_names
        )

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f"{self.model_name} - Prediction Explanation")
        plt.tight_layout()

        return fig, explanation

    def get_top_features(self, X, feature_names=None, top_n=10):
        """
        Get top contributing features for a prediction.

        Args:
            X: Single sample
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names, values, and SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")

        # Get SHAP values
        shap_values = self.explainer.shap_values(X.reshape(1, -1) if X.ndim == 1 else X)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values

        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(X.flatten()))]

        df = pd.DataFrame({
            'feature': feature_names,
            'value': X.flatten(),
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        })

        # Sort by absolute SHAP value
        df = df.sort_values('abs_shap', ascending=False).head(top_n)

        return df[['feature', 'value', 'shap_value']]

    def summary_plot(self, X, feature_names=None, max_display=20, save_path=None):
        """
        Create SHAP summary plot showing global feature importance.

        Args:
            X: Dataset (multiple samples)
            feature_names: List of feature names
            max_display: Maximum features to display
            save_path: Path to save figure (optional)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")

        # Get SHAP values for all samples
        shap_values = self.explainer.shap_values(X)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f"{self.model_name} - Feature Importance (SHAP)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def feature_importance_bar(self, X, feature_names=None, max_display=20, save_path=None):
        """
        Create bar plot of mean absolute SHAP values (feature importance).

        Args:
            X: Dataset
            feature_names: List of feature names
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_shap))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(max_display)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(df['feature'], df['importance'], color='steelblue')
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title(f"{self.model_name} - Feature Importance", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, df

    def dependence_plot(self, X, feature_idx, feature_names=None, interaction_idx='auto', save_path=None):
        """
        Create SHAP dependence plot for a single feature.

        Args:
            X: Dataset
            feature_idx: Index or name of feature to plot
            feature_names: List of feature names
            interaction_idx: Feature to use for coloring ('auto' for automatic selection)
            save_path: Path to save figure
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")

        # Get SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create dependence plot
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        plt.title(f"{self.model_name} - Dependence Plot")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()


def load_explainer_for_model(model_path, X_sample=None):
    """
    Load a trained model and create an explainer.

    Args:
        model_path: Path to .joblib model file
        X_sample: Sample data for background

    Returns:
        ModelExplainer instance
    """
    model = load(model_path)
    model_name = Path(model_path).stem.replace('_', ' ').title()

    return ModelExplainer(model, model_name, X_sample)


if __name__ == "__main__":
    # Example usage
    from src.config import DATA_PROCESSED, MODELS

    print("Loading test data...")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].values

    # Load best model
    print("\nLoading Gradient Boosting model...")
    explainer = load_explainer_for_model(MODELS / "gradient_boosting.joblib", X_test[:100])

    # Feature importance
    print("\nGenerating feature importance plot...")
    fig, importance_df = explainer.feature_importance_bar(
        X_test[:1000],
        max_display=15,
        save_path="reports/shap_feature_importance.png"
    )
    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    # Single prediction explanation
    print("\nExplaining a single attack prediction...")
    attack_samples = X_test[y_test == 1]
    attack_sample = attack_samples.iloc[0].values
    fig, explanation = explainer.explain_prediction(attack_sample, max_display=10)
    plt.savefig("reports/shap_waterfall_example.png", dpi=300, bbox_inches='tight')

    # Get top contributing features
    top_features = explainer.get_top_features(attack_sample, top_n=10)
    print("\nTop Contributing Features:")
    print(top_features.to_string(index=False))

    print("\n✓ SHAP analysis complete! Check reports/ for visualizations.")
