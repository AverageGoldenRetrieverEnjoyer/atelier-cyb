#!/usr/bin/env python3
"""
Evaluate and visualize results from unsupervised ML models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_isolation_forest_results(predictions_df: pd.DataFrame, output_dir: Path):
    """Visualize Isolation Forest results."""
    LOGGER.info("Creating Isolation Forest visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Anomaly score distribution
    ax = axes[0, 0]
    scores = predictions_df["isolation_forest_score"]
    ax.hist(scores, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(scores.quantile(0.1), color="red", linestyle="--", label="10th percentile")
    ax.set_xlabel("Anomaly Score (lower = more anomalous)")
    ax.set_ylabel("Frequency")
    ax.set_title("Isolation Forest: Anomaly Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Anomaly vs Normal counts
    ax = axes[0, 1]
    anomaly_counts = predictions_df["isolation_forest"].value_counts()
    labels = ["Normal (1)", "Anomaly (-1)"]
    colors = ["green", "red"]
    ax.pie(anomaly_counts.values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Isolation Forest: Anomaly Detection")
    
    # 3. Score vs prediction scatter
    ax = axes[1, 0]
    normal_scores = scores[predictions_df["isolation_forest"] == 1]
    anomaly_scores = scores[predictions_df["isolation_forest"] == -1]
    ax.scatter(range(len(normal_scores)), normal_scores, alpha=0.5, label="Normal", s=1)
    ax.scatter(range(len(normal_scores), len(normal_scores) + len(anomaly_scores)), 
               anomaly_scores, alpha=0.5, label="Anomaly", s=1, color="red")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Isolation Forest: Scores by Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Top anomalies (lowest scores)
    ax = axes[1, 1]
    top_anomalies = scores.nsmallest(100)
    ax.barh(range(len(top_anomalies)), top_anomalies.values)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Top 100 Anomalies (rank)")
    ax.set_title("Isolation Forest: Top 100 Anomalies")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(output_dir / "isolation_forest_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'isolation_forest_analysis.png'}")


def plot_kmeans_results(predictions_df: pd.DataFrame, X_scaled: np.ndarray, feature_names: List[str], output_dir: Path):
    """Visualize K-Means clustering results."""
    LOGGER.info("Creating K-Means visualizations...")
    
    # Load original data for PCA
    X_sample = X_scaled
    if len(X_scaled) > 10000:
        # Sample for visualization
        indices = np.random.choice(len(X_scaled), 10000, replace=False)
        X_sample = X_scaled[indices]
        labels_sample = predictions_df["kmeans_cluster"].iloc[indices].values
    else:
        labels_sample = predictions_df["kmeans_cluster"].values
    
    # PCA for 2D visualization
    LOGGER.info("Computing PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    # t-SNE for better separation (on smaller sample)
    if len(X_sample) <= 5000:
        LOGGER.info("Computing t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
    else:
        X_tsne = None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Cluster size distribution
    ax = axes[0, 0]
    cluster_counts = predictions_df["kmeans_cluster"].value_counts().sort_index()
    ax.bar(cluster_counts.index, cluster_counts.values, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Flows")
    ax.set_title("K-Means: Cluster Size Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    
    # 2. PCA visualization
    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sample, cmap="tab10", alpha=0.6, s=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("K-Means: Clusters in PCA Space")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    
    # 3. t-SNE visualization (if available)
    ax = axes[1, 0]
    if X_tsne is not None:
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, cmap="tab10", alpha=0.6, s=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("K-Means: Clusters in t-SNE Space")
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.text(0.5, 0.5, "t-SNE skipped\n(too many samples)", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("K-Means: t-SNE (skipped)")
    ax.grid(True, alpha=0.3)
    
    # 4. Cluster characteristics (mean values of key features)
    ax = axes[1, 1]
    # Select a few key features for visualization
    n_clusters = predictions_df["kmeans_cluster"].nunique()
    feature_display = feature_names[: min(10, len(feature_names))]
    if not feature_display:
        feature_display = [f"Feature_{i}" for i in range(min(10, X_scaled.shape[1]))]
    cluster_means = []
    for cluster_id in range(n_clusters):
        cluster_mask = predictions_df["kmeans_cluster"] == cluster_id
        cluster_data = X_scaled[cluster_mask]
        if len(cluster_data) == 0:
            cluster_means.append(np.zeros(len(feature_display)))
        else:
            cluster_means.append(cluster_data[:, : len(feature_display)].mean(axis=0))
    
    cluster_means = np.array(cluster_means)
    im = ax.imshow(cluster_means, aspect="auto", cmap="viridis")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster ID")
    ax.set_title("K-Means: Mean Feature Values per Cluster")
    ax.set_yticks(range(n_clusters))
    ax.set_xticks(range(len(feature_display)))
    ax.set_xticklabels(feature_display, rotation=45, ha="right")
    plt.colorbar(im, ax=ax, label="Mean Value")
    
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'kmeans_analysis.png'}")


def plot_autoencoder_results(predictions_df: pd.DataFrame, output_dir: Path):
    """Visualize Autoencoder results."""
    LOGGER.info("Creating Autoencoder visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reconstruction error distribution
    ax = axes[0, 0]
    errors = predictions_df["autoencoder_reconstruction_error"]
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    threshold = errors.quantile(0.95)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold (95th: {threshold:.4f})")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Autoencoder: Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Anomaly vs Normal counts
    ax = axes[0, 1]
    anomaly_counts = predictions_df["autoencoder"].value_counts()
    labels = ["Normal (1)", "Anomaly (-1)"]
    colors = ["green", "red"]
    ax.pie(anomaly_counts.values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Autoencoder: Anomaly Detection")
    
    # 3. Error vs prediction
    ax = axes[1, 0]
    normal_errors = errors[predictions_df["autoencoder"] == 1]
    anomaly_errors = errors[predictions_df["autoencoder"] == -1]
    ax.scatter(range(len(normal_errors)), normal_errors, alpha=0.5, label="Normal", s=1)
    ax.scatter(range(len(normal_errors), len(normal_errors) + len(anomaly_errors)), 
               anomaly_errors, alpha=0.5, label="Anomaly", s=1, color="red")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Threshold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Autoencoder: Errors by Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Top anomalies (highest errors)
    ax = axes[1, 1]
    top_anomalies = errors.nlargest(100)
    ax.barh(range(len(top_anomalies)), top_anomalies.values, color="red", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Top 100 Anomalies (rank)")
    ax.set_title("Autoencoder: Top 100 Anomalies")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(output_dir / "autoencoder_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'autoencoder_analysis.png'}")


def plot_feature_importance(if_model, feature_names: List[str], output_dir: Path):
    """Plot Isolation Forest feature importances."""
    if not hasattr(if_model, "feature_importances_"):
        LOGGER.warning("Isolation Forest model does not expose feature_importances_")
        return

    LOGGER.info("Creating feature importance plot...")
    importances = if_model.feature_importances_

    # Select top features
    top_n = min(20, len(importances))
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Isolation Forest Feature Importances (Top 20)")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'feature_importance.png'}")


def plot_feature_distributions(
    X_scaled: np.ndarray,
    predictions_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: Path,
    top_features: List[str],
):
    """Plot feature distributions for normal vs anomalies."""
    LOGGER.info("Creating feature distribution plots...")

    data_df = pd.DataFrame(X_scaled, columns=feature_names)
    data_df["IF_Anomaly"] = (predictions_df["isolation_forest"] == -1).astype(int)
    data_df["AE_Anomaly"] = (predictions_df["autoencoder"] == -1).astype(int)

    num_features = len(top_features)
    cols = 2
    rows = (num_features + 1) // cols
    plt.figure(figsize=(12, rows * 4))

    for idx, feature in enumerate(top_features):
        if feature not in data_df.columns:
            continue
        ax = plt.subplot(rows, cols, idx + 1)
        sns.kdeplot(
            data=data_df, x=feature, hue="IF_Anomaly",
            palette={0: "green", 1: "red"}, fill=True, common_norm=False, alpha=0.4, ax=ax
        )
        ax.set_title(f"Feature Distribution: {feature}")
        ax.set_xlabel("Scaled Value")
        ax.set_ylabel("Density")
        ax.legend(["Normal", "Anomaly"])
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'feature_distributions.png'}")


def plot_score_relationship(predictions_df: pd.DataFrame, output_dir: Path):
    """Plot relationship between IF scores and AE reconstruction errors."""
    LOGGER.info("Creating score relationship plot...")

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        predictions_df["isolation_forest_score"],
        predictions_df["autoencoder_reconstruction_error"],
        c=(predictions_df["isolation_forest"] == -1).astype(int),
        cmap="coolwarm",
        alpha=0.3,
        s=5,
    )
    plt.xlabel("Isolation Forest Score (lower = anomalous)")
    plt.ylabel("Autoencoder Reconstruction Error")
    plt.title("Isolation Forest vs Autoencoder Agreement")
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Normal", "Anomaly"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "if_vs_ae_scores.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'if_vs_ae_scores.png'}")


def plot_model_agreement(predictions_df: pd.DataFrame, output_dir: Path):
    """Visualize agreement between models."""
    LOGGER.info("Creating model agreement visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to binary (1 = anomaly, 0 = normal)
    if_anomaly = (predictions_df["isolation_forest"] == -1).astype(int)
    ae_anomaly = (predictions_df["autoencoder"] == -1).astype(int)
    
    # 1. Venn-like comparison (2-way)
    ax = axes[0, 0]
    both = ((if_anomaly == 1) & (ae_anomaly == 1)).sum()
    only_if = ((if_anomaly == 1) & (ae_anomaly == 0)).sum()
    only_ae = ((if_anomaly == 0) & (ae_anomaly == 1)).sum()
    neither = ((if_anomaly == 0) & (ae_anomaly == 0)).sum()
    
    categories = ["Both", "Only IF", "Only AE", "Neither"]
    counts = [both, only_if, only_ae, neither]
    colors = ["red", "orange", "purple", "green"]
    ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.7)
    ax.set_ylabel("Number of Flows")
    ax.set_title("Model Agreement: Isolation Forest vs Autoencoder")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count, f"{count:,}", ha="center", va="bottom")
    
    # 2. Correlation matrix
    ax = axes[0, 1]
    correlation_data = pd.DataFrame({
        "IF": if_anomaly,
        "AE": ae_anomaly,
        "KMeans": predictions_df["kmeans_cluster"],
    })
    corr_matrix = correlation_data.corr()
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title("Model Correlation Matrix")
    
    # Add text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Correlation")
    
    # 3. Agreement statistics
    ax = axes[1, 0]
    total = len(predictions_df)
    agreement_pct = (both / total * 100) if total > 0 else 0
    stats_text = f"""
    Total Flows: {total:,}
    
    Both Models Agree (Anomaly): {both:,} ({both/total*100:.1f}%)
    Only Isolation Forest: {only_if:,} ({only_if/total*100:.1f}%)
    Only Autoencoder: {only_ae:,} ({only_ae/total*100:.1f}%)
    Both Normal: {neither:,} ({neither/total*100:.1f}%)
    
    Agreement Rate: {agreement_pct:.1f}%
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", family="monospace")
    ax.axis("off")
    ax.set_title("Model Agreement Statistics")
    
    # 4. K-Means cluster distribution by anomaly detection
    ax = axes[1, 1]
    cluster_anomaly = pd.crosstab(
        predictions_df["kmeans_cluster"],
        if_anomaly,
        normalize="index"
    ) * 100
    cluster_anomaly.plot(kind="bar", ax=ax, stacked=True, color=["green", "red"], alpha=0.7)
    ax.set_xlabel("K-Means Cluster")
    ax.set_ylabel("Percentage")
    ax.set_title("Anomaly Detection Rate by Cluster")
    ax.legend(["Normal", "Anomaly (IF)"])
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_agreement.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'model_agreement.png'}")


def evaluate_models(
    predictions_path: Path,
    X_scaled_path: Path,
    preprocessor_path: Path,
    models_dir: Path,
    output_dir: Path,
) -> Dict:
    """Main evaluation function."""
    
    LOGGER.info("=" * 60)
    LOGGER.info("Evaluating Models")
    LOGGER.info("=" * 60)
    
    # Load predictions and data
    predictions_df = pd.read_csv(predictions_path)
    X_scaled_df = pd.read_csv(X_scaled_path)
    X_scaled = X_scaled_df.values
    
    LOGGER.info(f"Loaded {len(predictions_df):,} predictions")
    LOGGER.info(f"Data shape: {X_scaled.shape}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models and preprocessor metadata
    feature_names = X_scaled_df.columns.tolist()
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        feature_names = preprocessor.get("feature_names", feature_names)
    else:
        LOGGER.warning(f"Preprocessor file not found: {preprocessor_path}")

    if_model = None
    if models_dir.joinpath("isolation_forest.joblib").exists():
        if_model = joblib.load(models_dir / "isolation_forest.joblib")
    else:
        LOGGER.warning("Isolation Forest model not found; skipping feature importance plot")

    # Determine top features from IF importance if available
    if if_model is not None and hasattr(if_model, "feature_importances_"):
        importances = if_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][: min(6, len(importances))]
        top_features = [feature_names[i] for i in top_indices]
    else:
        top_features = feature_names[: min(6, len(feature_names))]

    # Create visualizations
    plot_isolation_forest_results(predictions_df, output_dir)
    plot_kmeans_results(predictions_df, X_scaled, feature_names, output_dir)
    plot_autoencoder_results(predictions_df, output_dir)
    plot_model_agreement(predictions_df, output_dir)
    if if_model is not None:
        plot_feature_importance(if_model, feature_names, output_dir)
    if top_features:
        plot_feature_distributions(X_scaled, predictions_df, feature_names, output_dir, top_features)
    plot_score_relationship(predictions_df, output_dir)
    
    # Calculate evaluation metrics
    if_anomalies = (predictions_df["isolation_forest"] == -1).sum()
    ae_anomalies = (predictions_df["autoencoder"] == -1).sum()
    n_clusters = predictions_df["kmeans_cluster"].nunique()
    
    evaluation = {
        "total_flows": len(predictions_df),
        "isolation_forest": {
            "anomalies_detected": int(if_anomalies),
            "anomaly_rate": float(if_anomalies / len(predictions_df)),
            "score_range": [
                float(predictions_df["isolation_forest_score"].min()),
                float(predictions_df["isolation_forest_score"].max()),
            ],
        },
        "kmeans": {
            "n_clusters": int(n_clusters),
            "cluster_sizes": predictions_df["kmeans_cluster"].value_counts().to_dict(),
        },
        "autoencoder": {
            "anomalies_detected": int(ae_anomalies),
            "anomaly_rate": float(ae_anomalies / len(predictions_df)),
            "threshold": float(predictions_df["autoencoder_reconstruction_error"].quantile(0.95)),
            "error_range": [
                float(predictions_df["autoencoder_reconstruction_error"].min()),
                float(predictions_df["autoencoder_reconstruction_error"].max()),
            ],
        },
        "model_agreement": {
            "both_anomaly": int(((predictions_df["isolation_forest"] == -1) & 
                                 (predictions_df["autoencoder"] == -1)).sum()),
            "agreement_rate": float(
                ((predictions_df["isolation_forest"] == predictions_df["autoencoder"]).sum() / len(predictions_df))
            ),
        },
    }
    
    # Save evaluation JSON
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    LOGGER.info("=" * 60)
    LOGGER.info("Evaluation Summary")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Isolation Forest: {if_anomalies:,} anomalies ({if_anomalies/len(predictions_df)*100:.2f}%)")
    LOGGER.info(f"K-Means: {n_clusters} clusters")
    LOGGER.info(f"Autoencoder: {ae_anomalies:,} anomalies ({ae_anomalies/len(predictions_df)*100:.2f}%)")
    LOGGER.info(f"Model Agreement: {evaluation['model_agreement']['agreement_rate']*100:.1f}%")
    LOGGER.info(f"Results saved to: {output_dir}")
    
    return evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize unsupervised ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions.csv")
    parser.add_argument("--X-scaled", type=Path, required=True, help="Path to X_scaled.csv")
    parser.add_argument("--preprocessor", type=Path, required=True, help="Path to preprocessor.joblib")
    parser.add_argument("--models-dir", type=Path, required=True, help="Directory containing trained models")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")

    args = parser.parse_args()

    evaluate_models(args.predictions, args.X_scaled, args.preprocessor, args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()

