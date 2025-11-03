"""
MLflow utilities for experiment tracking and model registry.
Tracks parameters, metrics, and artifacts for all model training runs.
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import json
from datetime import datetime


def setup_mlflow(experiment_name="intrusion-detection", tracking_uri="./mlruns"):
    """
    Setup MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: URI for MLflow tracking (local directory or remote server)
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_training(
    model,
    model_name,
    params,
    metrics,
    artifacts=None,
    tags=None
):
    """
    Log a complete model training run to MLflow.

    Args:
        model: Trained model object
        model_name: Name of the model (e.g., "gradient_boosting")
        params: Dictionary of hyperparameters
        metrics: Dictionary of evaluation metrics
        artifacts: Dictionary of artifact paths to log
        tags: Dictionary of tags for the run

    Returns:
        MLflow run ID
    """
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log model
        mlflow.sklearn.log_model(
            model,
            model_name,
            registered_model_name=f"{model_name}_model"
        )

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, artifact_path=name)

        # Get run ID
        run_id = mlflow.active_run().info.run_id

        print(f"✓ Logged {model_name} to MLflow (run_id: {run_id})")

        return run_id


def log_comparison_report(metrics_df, report_path, plots_dir):
    """
    Log model comparison results to MLflow.

    Args:
        metrics_df: DataFrame with model comparison metrics
        report_path: Path to comparison report markdown
        plots_dir: Directory with comparison plots
    """
    with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log summary metrics
        best_model = metrics_df['ROC AUC'].idxmax()
        best_auc = metrics_df.loc[best_model, 'ROC AUC']

        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_roc_auc", best_auc)
        mlflow.log_metric("num_models_compared", len(metrics_df))

        # Log individual model metrics
        for model_name, row in metrics_df.iterrows():
            mlflow.log_metrics({
                f"{model_name}_roc_auc": row['ROC AUC'],
                f"{model_name}_f1": row.get('F1 Score', 0),
                f"{model_name}_accuracy": row.get('Accuracy', 0),
            })

        # Log comparison report
        if Path(report_path).exists():
            mlflow.log_artifact(report_path, artifact_path="reports")

        # Log comparison plots
        if Path(plots_dir).exists():
            for plot_file in Path(plots_dir).glob("*.png"):
                mlflow.log_artifact(str(plot_file), artifact_path="plots")

        # Log metrics CSV
        csv_path = Path(plots_dir) / "model_comparison_metrics.csv"
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), artifact_path="data")

        run_id = mlflow.active_run().info.run_id
        print(f"✓ Logged model comparison to MLflow (run_id: {run_id})")

        return run_id


def register_best_model(model, model_name, metrics, stage="Staging"):
    """
    Register a model in MLflow Model Registry.

    Args:
        model: Trained model
        model_name: Name for the registered model
        metrics: Dictionary of model metrics
        stage: Stage to transition model to (None, "Staging", "Production", "Archived")

    Returns:
        Model version
    """
    # Log model with registry
    model_info = mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=model_name
    )

    # Transition to stage if specified
    if stage:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Get latest version
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            version = latest_versions[0].version

            # Transition to stage
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )

            print(f"✓ Model {model_name} version {version} transitioned to {stage}")
            return version

    return None


def load_model_from_registry(model_name, stage="Production"):
    """
    Load a model from MLflow Model Registry.

    Args:
        model_name: Name of the registered model
        stage: Stage to load from

    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def compare_runs(experiment_name="intrusion-detection", metric="roc_auc", top_n=10):
    """
    Compare runs in an experiment and return top performers.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        top_n: Number of top runs to return

    Returns:
        DataFrame with top runs
    """
    from mlflow.tracking import MlflowClient
    import pandas as pd

    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment {experiment_name} not found")
        return None

    # Search runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n
    )

    # Create DataFrame
    data = []
    for run in runs:
        data.append({
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'model': run.data.params.get('model_name', 'N/A'),
            metric: run.data.metrics.get(metric, 0),
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Example usage
    setup_mlflow()

    print("MLflow tracking server setup complete!")
    print("\nTo view the MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen visit: http://localhost:5000")
