#!/usr/bin/env python3
"""
Evaluate and visualize results from unsupervised ML models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

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


def plot_anomaly_alert_correlation(
    flows_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20
):
    """
    Plot correlation between detected anomalies and Suricata alerts.
    Shows which attack types the models recognize.
    
    Args:
        flows_df: DataFrame with flow information and alert correlation
        predictions_df: DataFrame with model predictions
        output_dir: Output directory for plots
        top_n: Number of top alert signatures to display
    """
    LOGGER.info("Creating anomaly-alert correlation visualizations...")
    
    if "has_alert" not in flows_df.columns or "alert_signatures" not in flows_df.columns:
        LOGGER.warning("Alert information not available in flows_df. Skipping anomaly-alert correlation plot.")
        return
    
    # Ensure flows_df and predictions_df have matching indices
    if len(flows_df) != len(predictions_df):
        LOGGER.warning(f"Flow count mismatch: {len(flows_df)} vs {len(predictions_df)}. Skipping alert correlation plot.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Get anomaly masks for each model
    if_anomalies = predictions_df["isolation_forest"] == -1
    ae_anomalies = predictions_df["autoencoder"] == -1
    both_anomalies = if_anomalies & ae_anomalies
    
    # Get anomaly flows for each model
    if_anomaly_flows = flows_df[if_anomalies]
    ae_anomaly_flows = flows_df[ae_anomalies]
    both_anomaly_flows = flows_df[both_anomalies]
    
    # Get flows with alerts
    if_anomaly_flows_with_alerts = if_anomaly_flows[if_anomaly_flows["has_alert"] == True] if len(if_anomaly_flows) > 0 else pd.DataFrame()
    ae_anomaly_flows_with_alerts = ae_anomaly_flows[ae_anomaly_flows["has_alert"] == True] if len(ae_anomaly_flows) > 0 else pd.DataFrame()
    both_anomaly_flows_with_alerts = both_anomaly_flows[both_anomaly_flows["has_alert"] == True] if len(both_anomaly_flows) > 0 else pd.DataFrame()
    
    # Helper function to extract and count alert signatures
    def extract_signatures(alert_signatures_str):
        """Extract individual signatures from the concatenated string."""
        if pd.isna(alert_signatures_str) or alert_signatures_str == "":
            return []
        # Split by " | " and take first signature if multiple
        signatures = str(alert_signatures_str).split(" | ")
        return [s.strip() for s in signatures if s.strip()]
    
    # 1. Top alert signatures for Isolation Forest anomalies
    ax1 = fig.add_subplot(gs[0, 0])
    
    if len(if_anomaly_flows_with_alerts) > 0:
        all_signatures = []
        for sig_str in if_anomaly_flows_with_alerts["alert_signatures"]:
            all_signatures.extend(extract_signatures(sig_str))
        
        if all_signatures:
            sig_counts = pd.Series(all_signatures).value_counts().head(top_n)
            ax1.barh(range(len(sig_counts)), sig_counts.values, color="coral", edgecolor="black")
            ax1.set_yticks(range(len(sig_counts)))
            ax1.set_yticklabels(sig_counts.index, fontsize=9)
            ax1.set_xlabel("Nombre d'anomalies", fontsize=11)
            ax1.set_title(f"Isolation Forest: Top {len(sig_counts)} Alertes Suricata\n({len(if_anomaly_flows_with_alerts):,} anomalies avec alertes)", 
                         fontsize=12, fontweight="bold")
            ax1.grid(True, alpha=0.3, axis="x")
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, "Aucune signature d'alerte disponible", 
                    ha="center", va="center", transform=ax1.transAxes, fontsize=12)
            ax1.set_title("Isolation Forest: Alertes Suricata", fontsize=12)
    else:
        ax1.text(0.5, 0.5, f"Aucune anomalie avec alerte\n({len(if_anomaly_flows):,} anomalies totales)", 
                ha="center", va="center", transform=ax1.transAxes, fontsize=12)
        ax1.set_title("Isolation Forest: Alertes Suricata", fontsize=12)
    
    # 2. Top alert signatures for Autoencoder anomalies
    ax2 = fig.add_subplot(gs[0, 1])
    
    if len(ae_anomaly_flows_with_alerts) > 0:
        all_signatures = []
        for sig_str in ae_anomaly_flows_with_alerts["alert_signatures"]:
            all_signatures.extend(extract_signatures(sig_str))
        
        if all_signatures:
            sig_counts = pd.Series(all_signatures).value_counts().head(top_n)
            ax2.barh(range(len(sig_counts)), sig_counts.values, color="lightblue", edgecolor="black")
            ax2.set_yticks(range(len(sig_counts)))
            ax2.set_yticklabels(sig_counts.index, fontsize=9)
            ax2.set_xlabel("Nombre d'anomalies", fontsize=11)
            ax2.set_title(f"Autoencoder: Top {len(sig_counts)} Alertes Suricata\n({len(ae_anomaly_flows_with_alerts):,} anomalies avec alertes)", 
                         fontsize=12, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="x")
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, "Aucune signature d'alerte disponible", 
                    ha="center", va="center", transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Autoencoder: Alertes Suricata", fontsize=12)
    else:
        ax2.text(0.5, 0.5, f"Aucune anomalie avec alerte\n({len(ae_anomaly_flows):,} anomalies totales)", 
                ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Autoencoder: Alertes Suricata", fontsize=12)
    
    # 3. Alert categories distribution for IF anomalies
    ax3 = fig.add_subplot(gs[1, 0])
    if "alert_categories" in if_anomaly_flows_with_alerts.columns and len(if_anomaly_flows_with_alerts) > 0:
        all_categories = []
        for cat_str in if_anomaly_flows_with_alerts["alert_categories"]:
            if pd.notna(cat_str) and cat_str != "":
                categories = str(cat_str).split(" | ")
                all_categories.extend([c.strip() for c in categories if c.strip()])
        
        if all_categories:
            cat_counts = pd.Series(all_categories).value_counts().head(15)
            ax3.barh(range(len(cat_counts)), cat_counts.values, color="coral", edgecolor="black", alpha=0.7)
            ax3.set_yticks(range(len(cat_counts)))
            ax3.set_yticklabels(cat_counts.index, fontsize=9)
            ax3.set_xlabel("Nombre d'anomalies", fontsize=11)
            ax3.set_title("Isolation Forest: Catégories d'Alertes", fontsize=12, fontweight="bold")
            ax3.grid(True, alpha=0.3, axis="x")
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, "Aucune catégorie disponible", 
                    ha="center", va="center", transform=ax3.transAxes, fontsize=12)
            ax3.set_title("Isolation Forest: Catégories d'Alertes", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "Aucune donnée disponible", 
                ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Isolation Forest: Catégories d'Alertes", fontsize=12)
    
    # 4. Alert categories distribution for AE anomalies
    ax4 = fig.add_subplot(gs[1, 1])
    if "alert_categories" in ae_anomaly_flows_with_alerts.columns and len(ae_anomaly_flows_with_alerts) > 0:
        all_categories = []
        for cat_str in ae_anomaly_flows_with_alerts["alert_categories"]:
            if pd.notna(cat_str) and cat_str != "":
                categories = str(cat_str).split(" | ")
                all_categories.extend([c.strip() for c in categories if c.strip()])
        
        if all_categories:
            cat_counts = pd.Series(all_categories).value_counts().head(15)
            ax4.barh(range(len(cat_counts)), cat_counts.values, color="lightblue", edgecolor="black", alpha=0.7)
            ax4.set_yticks(range(len(cat_counts)))
            ax4.set_yticklabels(cat_counts.index, fontsize=9)
            ax4.set_xlabel("Nombre d'anomalies", fontsize=11)
            ax4.set_title("Autoencoder: Catégories d'Alertes", fontsize=12, fontweight="bold")
            ax4.grid(True, alpha=0.3, axis="x")
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, "Aucune catégorie disponible", 
                    ha="center", va="center", transform=ax4.transAxes, fontsize=12)
            ax4.set_title("Autoencoder: Catégories d'Alertes", fontsize=12)
    else:
        ax4.text(0.5, 0.5, "Aucune donnée disponible", 
                ha="center", va="center", transform=ax4.transAxes, fontsize=12)
        ax4.set_title("Autoencoder: Catégories d'Alertes", fontsize=12)
    
    # 5. Comparison: Anomalies with vs without alerts
    ax5 = fig.add_subplot(gs[2, 0])
    if_with_alerts = if_anomaly_flows["has_alert"].sum()
    if_without_alerts = len(if_anomaly_flows) - if_with_alerts
    ae_with_alerts = ae_anomaly_flows["has_alert"].sum()
    ae_without_alerts = len(ae_anomaly_flows) - ae_with_alerts
    
    x = np.arange(2)
    width = 0.35
    ax5.bar(x - width/2, [if_with_alerts, ae_with_alerts], width, 
            label="Avec alertes", color="red", edgecolor="black", alpha=0.7)
    ax5.bar(x + width/2, [if_without_alerts, ae_without_alerts], width,
            label="Sans alertes", color="gray", edgecolor="black", alpha=0.7)
    ax5.set_xlabel("Modèle", fontsize=11)
    ax5.set_ylabel("Nombre d'anomalies", fontsize=11)
    ax5.set_title("Anomalies avec/sans Alertes Suricata", fontsize=12, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(["Isolation Forest", "Autoencoder"])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for i, (with_alert, without_alert) in enumerate([(if_with_alerts, if_without_alerts), 
                                                       (ae_with_alerts, ae_without_alerts)]):
        ax5.text(i - width/2, with_alert + max(if_with_alerts, ae_with_alerts) * 0.01, 
                f"{with_alert:,}", ha="center", va="bottom", fontsize=9)
        ax5.text(i + width/2, without_alert + max(if_without_alerts, ae_without_alerts) * 0.01,
                f"{without_alert:,}", ha="center", va="bottom", fontsize=9)
    
    # 6. Top signatures for anomalies detected by both models
    ax6 = fig.add_subplot(gs[2, 1])
    
    if len(both_anomaly_flows_with_alerts) > 0:
        all_signatures = []
        for sig_str in both_anomaly_flows_with_alerts["alert_signatures"]:
            all_signatures.extend(extract_signatures(sig_str))
        
        if all_signatures:
            sig_counts = pd.Series(all_signatures).value_counts().head(top_n)
            ax6.barh(range(len(sig_counts)), sig_counts.values, color="purple", edgecolor="black", alpha=0.7)
            ax6.set_yticks(range(len(sig_counts)))
            ax6.set_yticklabels(sig_counts.index, fontsize=9)
            ax6.set_xlabel("Nombre d'anomalies", fontsize=11)
            ax6.set_title(f"Modèles en Accord: Top {len(sig_counts)} Alertes\n({len(both_anomaly_flows_with_alerts):,} anomalies avec alertes)", 
                         fontsize=12, fontweight="bold")
            ax6.grid(True, alpha=0.3, axis="x")
            ax6.invert_yaxis()
        else:
            ax6.text(0.5, 0.5, "Aucune signature d'alerte disponible", 
                    ha="center", va="center", transform=ax6.transAxes, fontsize=12)
            ax6.set_title("Modèles en Accord: Alertes Suricata", fontsize=12)
    else:
        ax6.text(0.5, 0.5, f"Aucune anomalie avec alerte\n({len(both_anomaly_flows):,} anomalies totales)", 
                ha="center", va="center", transform=ax6.transAxes, fontsize=12)
        ax6.set_title("Modèles en Accord: Alertes Suricata", fontsize=12)
    
    plt.suptitle("Corrélation Anomalies Détectées / Alertes Suricata", 
                fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(output_dir / "anomaly_alert_correlation.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'anomaly_alert_correlation.png'}")


def load_suricata_alerts(eve_path: Path, show_progress: bool = False) -> pd.DataFrame:
    """Load Suricata alerts from eve.json file."""
    if not eve_path.exists():
        LOGGER.warning(f"EVE file not found: {eve_path}. Skipping alert correlation.")
        return pd.DataFrame()
    
    alerts = []
    LOGGER.info(f"Loading Suricata alerts from {eve_path}")
    
    try:
        with eve_path.open("r") as handle:
            iterator = tqdm(handle, desc="Loading alerts") if show_progress else handle
            for line in iterator:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event_type") == "alert":
                        alert_data = event.get("alert", {})
                        alerts.append({
                            "flow_id": event.get("flow_id"),
                            "timestamp": event.get("timestamp"),
                            "signature": alert_data.get("signature", ""),
                            "signature_id": alert_data.get("signature_id"),
                            "category": alert_data.get("category", ""),
                            "severity": alert_data.get("severity"),
                            "src_ip": event.get("src_ip"),
                            "dst_ip": event.get("dest_ip") or event.get("dst_ip"),
                            "src_port": event.get("src_port"),
                            "dst_port": event.get("dest_port"),
                            "proto": event.get("proto"),
                        })
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    LOGGER.debug(f"Skipping malformed alert: {e}")
                    continue
    except Exception as e:
        LOGGER.warning(f"Error loading alerts from {eve_path}: {e}")
        return pd.DataFrame()
    
    alerts_df = pd.DataFrame(alerts)
    if not alerts_df.empty:
        LOGGER.info(f"Loaded {len(alerts_df):,} alerts")
    return alerts_df


def correlate_flows_with_alerts(
    flows_df: pd.DataFrame, 
    alerts_df: pd.DataFrame,
    time_window_seconds: float = 5.0
) -> pd.DataFrame:
    """
    Correlate flows with Suricata alerts based on IPs, ports, protocol, and timestamps.
    Uses vectorized operations for memory efficiency and speed.
    
    Args:
        flows_df: DataFrame with flow information (src_ip, dst_ip, src_port, dst_port, protocol, first_seen, last_seen)
        alerts_df: DataFrame with Suricata alerts
        time_window_seconds: Time window in seconds for matching timestamps
    
    Returns:
        DataFrame with flows and correlated alert information
    """
    if alerts_df.empty:
        LOGGER.warning("No alerts to correlate")
        flows_df["alert_count"] = 0
        flows_df["has_alert"] = False
        flows_df["alert_signatures"] = ""
        flows_df["max_alert_severity"] = np.nan
        flows_df["alert_categories"] = ""
        return flows_df
    
    LOGGER.info("Correlating flows with alerts (vectorized)...")
    
    # Make a copy to avoid modifying original
    flows_df = flows_df.copy()
    alerts_df = alerts_df.copy()
    
    # Convert timestamps to datetime (ensure both are timezone-naive)
    if "first_seen" in flows_df.columns:
        flows_df["first_seen_dt"] = pd.to_datetime(flows_df["first_seen"], errors="coerce", unit="ms")
        flows_df["last_seen_dt"] = pd.to_datetime(flows_df["last_seen"], errors="coerce", unit="ms")
        # Remove timezone if present
        for col in ["first_seen_dt", "last_seen_dt"]:
            if col in flows_df.columns and flows_df[col].dtype.name.startswith("datetime64"):
                try:
                    if flows_df[col].dt.tz is not None:
                        flows_df[col] = flows_df[col].dt.tz_convert("UTC").dt.tz_localize(None)
                except (AttributeError, TypeError):
                    pass
    
    if "timestamp" in alerts_df.columns:
        alerts_df["timestamp_dt"] = pd.to_datetime(alerts_df["timestamp"], errors="coerce")
        if alerts_df["timestamp_dt"].dtype.name.startswith("datetime64"):
            try:
                if alerts_df["timestamp_dt"].dt.tz is not None:
                    alerts_df["timestamp_dt"] = alerts_df["timestamp_dt"].dt.tz_convert("UTC").dt.tz_localize(None)
            except (AttributeError, TypeError):
                pass
    
    # Initialize result columns
    flows_df["alert_count"] = 0
    flows_df["has_alert"] = False
    flows_df["alert_signatures"] = ""
    flows_df["max_alert_severity"] = np.nan
    flows_df["alert_categories"] = ""
    
    # Prepare flows for matching - create both directions (normal and reversed)
    # This handles bidirectional flows (alerts can match either direction)
    flow_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "first_seen_dt", "last_seen_dt"]
    flows_normal = flows_df[flow_cols].copy()
    flows_normal["flow_idx"] = flows_normal.index
    
    # Create reversed version (swap src/dst) - use direct assignment
    flows_reversed = flows_df[flow_cols].copy()
    flows_reversed["flow_idx"] = flows_reversed.index
    # Swap IPs and ports for reversed direction
    temp_src_ip = flows_reversed["src_ip"].copy()
    temp_src_port = flows_reversed["src_port"].copy()
    flows_reversed["src_ip"] = flows_reversed["dst_ip"]
    flows_reversed["dst_ip"] = temp_src_ip
    flows_reversed["src_port"] = flows_reversed["dst_port"]
    flows_reversed["dst_port"] = temp_src_port
    
    # Combine both directions
    flows_for_matching = pd.concat([flows_normal, flows_reversed], ignore_index=True)
    
    # Prepare alerts for matching
    alerts_for_matching = alerts_df[["src_ip", "dst_ip", "src_port", "dst_port", "proto", "timestamp", "timestamp_dt", "signature", "category", "severity"]].copy()
    
    # Ensure protocol column names match
    if "protocol" in flows_for_matching.columns and "proto" in alerts_for_matching.columns:
        flows_for_matching.rename(columns={"protocol": "proto"}, inplace=True)
    
    # Step 1: Match on IPs and ports (vectorized)
    # First merge on exact IP/port matches
    LOGGER.info("Step 1/3: Matching on IPs and ports...")
    merged = flows_for_matching.merge(
        alerts_for_matching,
        on=["src_ip", "dst_ip", "src_port", "dst_port", "proto"],
        how="inner",
        suffixes=("_flow", "_alert")
    )
    
    if len(merged) == 0:
        LOGGER.info("No matches found on IP/port/protocol")
        # Clean up temporary columns
        if "first_seen_dt" in flows_df.columns:
            flows_df = flows_df.drop(columns=["first_seen_dt", "last_seen_dt"])
        return flows_df
    
    LOGGER.info(f"Found {len(merged):,} potential matches after IP/port/protocol matching")
    
    # Step 2: Filter by time window (vectorized)
    LOGGER.info("Step 2/3: Filtering by time window...")
    if "first_seen_dt" in merged.columns and "timestamp_dt" in merged.columns:
        time_window = pd.Timedelta(seconds=time_window_seconds)
        time_mask = (
            merged["timestamp_dt"].notna() &
            merged["first_seen_dt"].notna() &
            merged["last_seen_dt"].notna() &
            (merged["timestamp_dt"] >= merged["first_seen_dt"] - time_window) &
            (merged["timestamp_dt"] <= merged["last_seen_dt"] + time_window)
        )
        merged = merged[time_mask]
        LOGGER.info(f"Found {len(merged):,} matches after time filtering")
    
    if len(merged) == 0:
        LOGGER.info("No matches found after time filtering")
        # Clean up temporary columns
        if "first_seen_dt" in flows_df.columns:
            flows_df = flows_df.drop(columns=["first_seen_dt", "last_seen_dt"])
        return flows_df
    
    # Step 3: Aggregate alerts per flow (vectorized groupby)
    LOGGER.info("Step 3/3: Aggregating alerts per flow...")
    
    # Group by flow index and aggregate
    alert_agg = merged.groupby("flow_idx").agg({
        "signature": lambda x: " | ".join(x.dropna().unique()[:5]),  # Limit to 5 signatures
        "category": lambda x: " | ".join(x.dropna().unique()[:5]),  # Limit to 5 categories
        "severity": "max" if "severity" in merged.columns else lambda x: np.nan,
    }).reset_index()
    
    alert_agg.rename(columns={
        "signature": "alert_signatures",
        "category": "alert_categories",
        "severity": "max_alert_severity"
    }, inplace=True)
    
    # Count alerts per flow
    alert_counts = merged.groupby("flow_idx").size().reset_index(name="alert_count")
    
    # Merge aggregations
    alert_summary = alert_counts.merge(alert_agg, on="flow_idx", how="left")
    
    # Update flows_df with alert information (vectorized)
    flows_df.loc[alert_summary["flow_idx"], "alert_count"] = alert_summary["alert_count"].values
    flows_df.loc[alert_summary["flow_idx"], "has_alert"] = True
    flows_df.loc[alert_summary["flow_idx"], "alert_signatures"] = alert_summary["alert_signatures"].fillna("").values
    flows_df.loc[alert_summary["flow_idx"], "alert_categories"] = alert_summary["alert_categories"].fillna("").values
    if "max_alert_severity" in alert_summary.columns:
        flows_df.loc[alert_summary["flow_idx"], "max_alert_severity"] = alert_summary["max_alert_severity"].values
    
    matched_count = (flows_df["has_alert"] == True).sum()
    LOGGER.info(f"Matched {matched_count:,} flows ({matched_count/len(flows_df)*100:.1f}%) with alerts")
    
    # Clean up temporary columns
    if "first_seen_dt" in flows_df.columns:
        flows_df = flows_df.drop(columns=["first_seen_dt", "last_seen_dt"])
    
    return flows_df


def save_anomalous_flows_by_model(
    flows_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    prediction_column: str,
    score_column: Optional[str] = None
):
    """
    Save anomalous flows detected by a specific model to CSV.
    Includes alert correlation information if available.
    
    Args:
        flows_df: DataFrame with flow information and alert correlation
        predictions_df: DataFrame with model predictions
        output_dir: Output directory
        model_name: Name of the model (e.g., "isolation_forest", "autoencoder", "kmeans")
        prediction_column: Column name in predictions_df with predictions (-1 = anomaly, 1 = normal)
        score_column: Optional column name with anomaly scores
    """
    # Get anomalous flows (prediction == -1)
    anomaly_mask = predictions_df[prediction_column] == -1
    anomalous_indices = predictions_df[anomaly_mask].index
    
    if len(anomalous_indices) == 0:
        LOGGER.warning(f"No anomalies detected by {model_name}")
        return
    
    # Get corresponding flows
    anomalous_flows = flows_df.iloc[anomalous_indices].copy()
    
    # Add prediction information
    anomalous_flows["model"] = model_name
    anomalous_flows["prediction"] = predictions_df.loc[anomalous_indices, prediction_column].values
    
    if score_column and score_column in predictions_df.columns:
        anomalous_flows["anomaly_score"] = predictions_df.loc[anomalous_indices, score_column].values
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_dir / f"anomalous_flows_{model_name}.csv"
    anomalous_flows.to_csv(output_file, index=False)
    LOGGER.info(f"Saved {len(anomalous_flows):,} anomalous flows from {model_name} to {output_file}")
    
    # Log alert correlation statistics if available
    if "has_alert" in anomalous_flows.columns:
        flows_with_alerts = anomalous_flows["has_alert"].sum()
        LOGGER.info(f"  - {flows_with_alerts:,} flows ({flows_with_alerts/len(anomalous_flows)*100:.1f}%) have associated Suricata alerts")


def plot_feature_correlation_matrix(
    X_scaled: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    max_features: int = 30
):
    """Plot correlation matrix heatmap of features."""
    LOGGER.info("Creating feature correlation matrix...")
    
    # Limit number of features if too many
    if len(feature_names) > max_features:
        LOGGER.info(f"Limiting to top {max_features} features for correlation matrix")
        # Use first max_features features
        X_subset = X_scaled[:, :max_features]
        feature_subset = feature_names[:max_features]
    else:
        X_subset = X_scaled
        feature_subset = feature_names
    
    # Create DataFrame and compute correlation
    df = pd.DataFrame(X_subset, columns=feature_subset)
    corr_matrix = df.corr()
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Mask upper triangle
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        annot_kws={"size": 7}
    )
    
    plt.title("Matrice de Corrélation des Features", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'feature_correlation_matrix.png'}")


def plot_anomaly_feature_comparison(
    X_scaled: np.ndarray,
    predictions_df: pd.DataFrame,
    feature_names: List[str],
    output_dir: Path,
    top_n_features: int = 15
):
    """Compare feature values between normal and anomalous flows."""
    LOGGER.info("Creating anomaly vs normal feature comparison...")
    
    # Get anomaly masks
    if_anomalies = predictions_df["isolation_forest"] == -1
    ae_anomalies = predictions_df["autoencoder"] == -1
    both_anomalies = if_anomalies & ae_anomalies
    
    # Limit to top features
    n_features = min(top_n_features, len(feature_names), X_scaled.shape[1])
    feature_subset = feature_names[:n_features]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Isolation Forest: Mean feature values comparison
    ax = axes[0, 0]
    normal_means = X_scaled[~if_anomalies, :n_features].mean(axis=0)
    anomaly_means = X_scaled[if_anomalies, :n_features].mean(axis=0)
    
    x = np.arange(n_features)
    width = 0.35
    ax.bar(x - width/2, normal_means, width, label="Normal", color="green", alpha=0.7, edgecolor="black")
    ax.bar(x + width/2, anomaly_means, width, label="Anomalie (IF)", color="red", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Valeur Moyenne (normalisée)", fontsize=11)
    ax.set_title("Isolation Forest: Comparaison Features Normal vs Anomalie", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_subset, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # 2. Autoencoder: Mean feature values comparison
    ax = axes[0, 1]
    normal_means_ae = X_scaled[~ae_anomalies, :n_features].mean(axis=0)
    anomaly_means_ae = X_scaled[ae_anomalies, :n_features].mean(axis=0)
    
    ax.bar(x - width/2, normal_means_ae, width, label="Normal", color="green", alpha=0.7, edgecolor="black")
    ax.bar(x + width/2, anomaly_means_ae, width, label="Anomalie (AE)", color="purple", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Valeur Moyenne (normalisée)", fontsize=11)
    ax.set_title("Autoencoder: Comparaison Features Normal vs Anomalie", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_subset, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # 3. Feature difference (anomaly - normal) for IF
    ax = axes[1, 0]
    diff_if = anomaly_means - normal_means
    colors = ["red" if d > 0 else "blue" for d in diff_if]
    ax.barh(range(n_features), diff_if, color=colors, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Différence (Anomalie - Normal)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title("Isolation Forest: Différence des Features", fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_subset, fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    
    # 4. Feature difference (anomaly - normal) for AE
    ax = axes[1, 1]
    diff_ae = anomaly_means_ae - normal_means_ae
    colors_ae = ["purple" if d > 0 else "blue" for d in diff_ae]
    ax.barh(range(n_features), diff_ae, color=colors_ae, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Différence (Anomalie - Normal)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title("Autoencoder: Différence des Features", fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_subset, fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.suptitle("Analyse Comparative des Features: Normal vs Anomalies", 
                fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_feature_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'anomaly_feature_comparison.png'}")


def plot_temporal_anomaly_distribution(
    flows_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path
):
    """Plot temporal distribution of anomalies if timestamp data is available."""
    LOGGER.info("Creating temporal anomaly distribution...")
    
    if flows_df is None or len(flows_df) != len(predictions_df):
        LOGGER.warning("Cannot create temporal plot: flows_df not available or size mismatch")
        return
    
    # Check for timestamp columns
    timestamp_cols = ["first_seen", "last_seen", "timestamp", "start_time", "end_time"]
    available_cols = [col for col in timestamp_cols if col in flows_df.columns]
    
    if not available_cols:
        LOGGER.warning("No timestamp columns found for temporal analysis")
        return
    
    # Use first available timestamp column
    time_col = available_cols[0]
    
    try:
        # Convert to datetime
        if time_col in ["first_seen", "last_seen"]:
            flows_df["time_dt"] = pd.to_datetime(flows_df[time_col], errors="coerce", unit="ms")
        else:
            flows_df["time_dt"] = pd.to_datetime(flows_df[time_col], errors="coerce")
        
        # Remove invalid timestamps
        valid_mask = flows_df["time_dt"].notna()
        flows_valid = flows_df[valid_mask].copy()
        predictions_valid = predictions_df[valid_mask].copy()
        
        if len(flows_valid) == 0:
            LOGGER.warning("No valid timestamps found")
            return
        
        # Create time bins (hourly)
        flows_valid["hour"] = flows_valid["time_dt"].dt.floor("H")
        
        # Get anomaly masks
        if_anomalies = (predictions_valid["isolation_forest"] == -1).astype(int)
        ae_anomalies = (predictions_valid["autoencoder"] == -1).astype(int)
        
        # Aggregate by hour
        hourly_stats = flows_valid.groupby("hour").agg({
            "time_dt": "count"  # Count flows per hour
        }).rename(columns={"time_dt": "total_flows"})
        
        hourly_stats["if_anomalies"] = flows_valid.groupby("hour")[if_anomalies].sum()
        hourly_stats["ae_anomalies"] = flows_valid.groupby("hour")[ae_anomalies].sum()
        hourly_stats["if_anomaly_rate"] = (hourly_stats["if_anomalies"] / hourly_stats["total_flows"] * 100)
        hourly_stats["ae_anomaly_rate"] = (hourly_stats["ae_anomalies"] / hourly_stats["total_flows"] * 100)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. Total flows over time
        ax = axes[0]
        ax.plot(hourly_stats.index, hourly_stats["total_flows"], marker="o", linewidth=2, markersize=4, color="blue")
        ax.set_xlabel("Temps", fontsize=11)
        ax.set_ylabel("Nombre de Flows", fontsize=11)
        ax.set_title("Distribution Temporelle: Volume de Trafic", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # 2. Anomaly counts over time
        ax = axes[1]
        ax.plot(hourly_stats.index, hourly_stats["if_anomalies"], marker="o", linewidth=2, 
                markersize=4, label="Isolation Forest", color="red", alpha=0.7)
        ax.plot(hourly_stats.index, hourly_stats["ae_anomalies"], marker="s", linewidth=2, 
                markersize=4, label="Autoencoder", color="purple", alpha=0.7)
        ax.set_xlabel("Temps", fontsize=11)
        ax.set_ylabel("Nombre d'Anomalies", fontsize=11)
        ax.set_title("Distribution Temporelle: Anomalies Détectées", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # 3. Anomaly rate over time
        ax = axes[2]
        ax.plot(hourly_stats.index, hourly_stats["if_anomaly_rate"], marker="o", linewidth=2, 
                markersize=4, label="Isolation Forest", color="red", alpha=0.7)
        ax.plot(hourly_stats.index, hourly_stats["ae_anomaly_rate"], marker="s", linewidth=2, 
                markersize=4, label="Autoencoder", color="purple", alpha=0.7)
        ax.set_xlabel("Temps", fontsize=11)
        ax.set_ylabel("Taux d'Anomalies (%)", fontsize=11)
        ax.set_title("Distribution Temporelle: Taux d'Anomalies", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.suptitle("Analyse Temporelle des Anomalies", fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / "temporal_anomaly_distribution.png", dpi=200, bbox_inches="tight")
        plt.close()
        LOGGER.info(f"Saved: {output_dir / 'temporal_anomaly_distribution.png'}")
        
    except Exception as e:
        LOGGER.warning(f"Error creating temporal plot: {e}")


def plot_network_statistics(
    flows_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path
):
    """Plot network statistics for normal vs anomalous flows."""
    LOGGER.info("Creating network statistics visualizations...")
    
    if flows_df is None or len(flows_df) != len(predictions_df):
        LOGGER.warning("Cannot create network statistics: flows_df not available or size mismatch")
        return
    
    # Get anomaly masks
    if_anomalies = predictions_df["isolation_forest"] == -1
    ae_anomalies = predictions_df["autoencoder"] == -1
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Protocol distribution for IF anomalies
    ax = axes[0, 0]
    if "protocol" in flows_df.columns:
        normal_protocols = flows_df[~if_anomalies]["protocol"].value_counts().head(10)
        anomaly_protocols = flows_df[if_anomalies]["protocol"].value_counts().head(10)
        
        x = np.arange(len(normal_protocols))
        width = 0.35
        ax.bar(x - width/2, normal_protocols.values, width, label="Normal", 
               color="green", alpha=0.7, edgecolor="black")
        ax.bar(x + width/2, anomaly_protocols.values, width, label="Anomalie (IF)", 
               color="red", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Protocole", fontsize=11)
        ax.set_ylabel("Nombre de Flows", fontsize=11)
        ax.set_title("Isolation Forest: Distribution par Protocole", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(normal_protocols.index, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Colonne 'protocol' non disponible", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Distribution par Protocole", fontsize=12)
    
    # 2. Top source IPs for anomalies
    ax = axes[0, 1]
    if "src_ip" in flows_df.columns:
        anomaly_ips = flows_df[if_anomalies]["src_ip"].value_counts().head(10)
        if len(anomaly_ips) > 0:
            ax.barh(range(len(anomaly_ips)), anomaly_ips.values, color="red", alpha=0.7, edgecolor="black")
            ax.set_yticks(range(len(anomaly_ips)))
            ax.set_yticklabels(anomaly_ips.index, fontsize=9)
            ax.set_xlabel("Nombre d'Anomalies", fontsize=11)
            ax.set_title("Top 10 IPs Sources (Isolation Forest)", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "Aucune anomalie trouvée", 
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, "Colonne 'src_ip' non disponible", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Top IPs Sources", fontsize=12)
    
    # 3. Port distribution for anomalies
    ax = axes[1, 0]
    if "dst_port" in flows_df.columns:
        normal_ports = flows_df[~if_anomalies]["dst_port"].value_counts().head(15)
        anomaly_ports = flows_df[if_anomalies]["dst_port"].value_counts().head(15)
        
        # Combine and get top ports
        all_ports = set(normal_ports.index) | set(anomaly_ports.index)
        top_ports = sorted(list(all_ports))[:15]
        
        normal_counts = [normal_ports.get(p, 0) for p in top_ports]
        anomaly_counts = [anomaly_ports.get(p, 0) for p in top_ports]
        
        x = np.arange(len(top_ports))
        width = 0.35
        ax.bar(x - width/2, normal_counts, width, label="Normal", 
               color="green", alpha=0.7, edgecolor="black")
        ax.bar(x + width/2, anomaly_counts, width, label="Anomalie (IF)", 
               color="red", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Port Destination", fontsize=11)
        ax.set_ylabel("Nombre de Flows", fontsize=11)
        ax.set_title("Isolation Forest: Distribution par Port Destination", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(top_ports, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Colonne 'dst_port' non disponible", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Distribution par Port", fontsize=12)
    
    # 4. Flow duration comparison
    ax = axes[1, 1]
    duration_cols = ["duration", "flow_duration", "last_seen", "first_seen"]
    duration_col = None
    for col in duration_cols:
        if col in flows_df.columns:
            duration_col = col
            break
    
    if duration_col and duration_col in ["last_seen", "first_seen"]:
        # Calculate duration
        if "first_seen" in flows_df.columns and "last_seen" in flows_df.columns:
            flows_df["duration_ms"] = flows_df["last_seen"] - flows_df["first_seen"]
            normal_durations = flows_df[~if_anomalies]["duration_ms"] / 1000  # Convert to seconds
            anomaly_durations = flows_df[if_anomalies]["duration_ms"] / 1000
            
            ax.hist(normal_durations, bins=50, alpha=0.6, label="Normal", color="green", edgecolor="black")
            ax.hist(anomaly_durations, bins=50, alpha=0.6, label="Anomalie (IF)", color="red", edgecolor="black")
            ax.set_xlabel("Durée du Flow (secondes)", fontsize=11)
            ax.set_ylabel("Fréquence", fontsize=11)
            ax.set_title("Distribution de la Durée des Flows", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, normal_durations.quantile(0.99))  # Limit to 99th percentile for visibility
    elif duration_col:
        normal_durations = flows_df[~if_anomalies][duration_col]
        anomaly_durations = flows_df[if_anomalies][duration_col]
        
        ax.hist(normal_durations, bins=50, alpha=0.6, label="Normal", color="green", edgecolor="black")
        ax.hist(anomaly_durations, bins=50, alpha=0.6, label="Anomalie (IF)", color="red", edgecolor="black")
        ax.set_xlabel(f"Durée ({duration_col})", fontsize=11)
        ax.set_ylabel("Fréquence", fontsize=11)
        ax.set_title("Distribution de la Durée des Flows", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Colonne de durée non disponible", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Distribution de Durée", fontsize=12)
    
    plt.suptitle("Statistiques Réseau: Normal vs Anomalies", 
                fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "network_statistics.png", dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"Saved: {output_dir / 'network_statistics.png'}")


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
    original_flows_path: Optional[Path] = None,
    eve_json_path: Optional[Path] = None,
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

    # Load original flows and correlate with alerts if provided
    flows_df = None
    if original_flows_path and original_flows_path.exists():
        LOGGER.info(f"Loading original flows from {original_flows_path}")
        flows_df = pd.read_csv(original_flows_path)
        LOGGER.info(f"Loaded {len(flows_df):,} original flows")
        
        # Ensure flows_df has the same number of rows as predictions_df
        if len(flows_df) != len(predictions_df):
            LOGGER.warning(f"Flow count mismatch: {len(flows_df)} flows vs {len(predictions_df)} predictions")
            # If flows_df is longer, truncate to match predictions
            if len(flows_df) > len(predictions_df):
                flows_df = flows_df.iloc[:len(predictions_df)].reset_index(drop=True)
                LOGGER.info(f"Truncated flows to {len(flows_df):,} to match predictions")
            # If flows_df is shorter, we can't proceed with correlation
            elif len(flows_df) < len(predictions_df):
                LOGGER.error("Cannot correlate: fewer flows than predictions")
                flows_df = None
        
        # Load and correlate alerts if eve.json is provided
        if flows_df is not None and eve_json_path and eve_json_path.exists():
            LOGGER.info(f"Loading Suricata alerts from {eve_json_path}")
            alerts_df = load_suricata_alerts(eve_json_path, show_progress=True)
            
            if not alerts_df.empty:
                LOGGER.info("Correlating flows with Suricata alerts...")
                flows_df = correlate_flows_with_alerts(flows_df, alerts_df, time_window_seconds=5.0)
            else:
                LOGGER.warning("No alerts found in eve.json")
                # Initialize alert columns anyway
                flows_df["alert_count"] = 0
                flows_df["has_alert"] = False
                flows_df["alert_signatures"] = ""
                flows_df["max_alert_severity"] = np.nan
                flows_df["alert_categories"] = ""
        elif flows_df is not None:
            # Initialize alert columns if no eve.json provided
            flows_df["alert_count"] = 0
            flows_df["has_alert"] = False
            flows_df["alert_signatures"] = ""
            flows_df["max_alert_severity"] = np.nan
            flows_df["alert_categories"] = ""
    else:
        if original_flows_path:
            LOGGER.warning(f"Original flows file not found: {original_flows_path}")
        else:
            LOGGER.info("No original flows path provided; skipping alert correlation")

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
    
    # New visualizations
    plot_feature_correlation_matrix(X_scaled, feature_names, output_dir, max_features=30)
    plot_anomaly_feature_comparison(X_scaled, predictions_df, feature_names, output_dir, top_n_features=15)
    
    # Plot anomaly-alert correlation if flows and alerts are available
    if flows_df is not None and len(flows_df) == len(predictions_df):
        plot_anomaly_alert_correlation(flows_df, predictions_df, output_dir, top_n=20)
        plot_temporal_anomaly_distribution(flows_df, predictions_df, output_dir)
        plot_network_statistics(flows_df, predictions_df, output_dir)
    
    # Save anomalous flows by model if flows_df is available
    if flows_df is not None:
        LOGGER.info("=" * 60)
        LOGGER.info("Saving Anomalous Flows by Model")
        LOGGER.info("=" * 60)
        
        # Create directory for anomalous flows
        anomalous_flows_dir = output_dir / "anomalous_flows"
        anomalous_flows_dir.mkdir(parents=True, exist_ok=True)
        
        # Save for each model
        save_anomalous_flows_by_model(
            flows_df, predictions_df, anomalous_flows_dir,
            "isolation_forest", "isolation_forest", "isolation_forest_score"
        )
        save_anomalous_flows_by_model(
            flows_df, predictions_df, anomalous_flows_dir,
            "autoencoder", "autoencoder", "autoencoder_reconstruction_error"
        )
        
        # For K-Means, we'll save flows from the smallest cluster(s) as anomalies
        # Typically, smaller clusters are more likely to be anomalies
        cluster_sizes = predictions_df["kmeans_cluster"].value_counts()
        if len(cluster_sizes) > 1:
            # Consider smallest cluster as anomaly
            smallest_cluster = cluster_sizes.idxmin()
            kmeans_anomaly_mask = predictions_df["kmeans_cluster"] == smallest_cluster
            kmeans_anomaly_indices = predictions_df[kmeans_anomaly_mask].index
            
            if len(kmeans_anomaly_indices) > 0:
                kmeans_anomalous_flows = flows_df.iloc[kmeans_anomaly_indices].copy()
                kmeans_anomalous_flows["model"] = "kmeans"
                kmeans_anomalous_flows["cluster"] = predictions_df.loc[kmeans_anomaly_indices, "kmeans_cluster"].values
                
                output_file = anomalous_flows_dir / "anomalous_flows_kmeans.csv"
                kmeans_anomalous_flows.to_csv(output_file, index=False)
                LOGGER.info(f"Saved {len(kmeans_anomalous_flows):,} flows from smallest K-Means cluster to {output_file}")
                
                if "has_alert" in kmeans_anomalous_flows.columns:
                    flows_with_alerts = kmeans_anomalous_flows["has_alert"].sum()
                    LOGGER.info(f"  - {flows_with_alerts:,} flows ({flows_with_alerts/len(kmeans_anomalous_flows)*100:.1f}%) have associated Suricata alerts")
    
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
    
    # Add alert correlation statistics if available
    if flows_df is not None and "has_alert" in flows_df.columns:
        total_with_alerts = flows_df["has_alert"].sum()
        if_anomalies_with_alerts = flows_df.loc[predictions_df["isolation_forest"] == -1, "has_alert"].sum() if len(flows_df) == len(predictions_df) else 0
        ae_anomalies_with_alerts = flows_df.loc[predictions_df["autoencoder"] == -1, "has_alert"].sum() if len(flows_df) == len(predictions_df) else 0
        
        evaluation["alert_correlation"] = {
            "total_flows_with_alerts": int(total_with_alerts),
            "if_anomalies_with_alerts": int(if_anomalies_with_alerts),
            "ae_anomalies_with_alerts": int(ae_anomalies_with_alerts),
            "if_anomaly_alert_rate": float(if_anomalies_with_alerts / if_anomalies) if if_anomalies > 0 else 0.0,
            "ae_anomaly_alert_rate": float(ae_anomalies_with_alerts / ae_anomalies) if ae_anomalies > 0 else 0.0,
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
    
    if flows_df is not None and "has_alert" in flows_df.columns:
        total_with_alerts = flows_df["has_alert"].sum()
        LOGGER.info(f"Alert Correlation: {total_with_alerts:,} flows ({total_with_alerts/len(flows_df)*100:.1f}%) have Suricata alerts")
        if_anomalies_with_alerts = flows_df.loc[predictions_df["isolation_forest"] == -1, "has_alert"].sum() if len(flows_df) == len(predictions_df) else 0
        ae_anomalies_with_alerts = flows_df.loc[predictions_df["autoencoder"] == -1, "has_alert"].sum() if len(flows_df) == len(predictions_df) else 0
        LOGGER.info(f"  - IF anomalies with alerts: {if_anomalies_with_alerts:,} ({if_anomalies_with_alerts/if_anomalies*100:.1f}%)" if if_anomalies > 0 else "  - IF anomalies with alerts: 0")
        LOGGER.info(f"  - AE anomalies with alerts: {ae_anomalies_with_alerts:,} ({ae_anomalies_with_alerts/ae_anomalies*100:.1f}%)" if ae_anomalies > 0 else "  - AE anomalies with alerts: 0")
    
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
    parser.add_argument("--original-flows", type=Path, default=None, help="Path to original_flows.csv (for alert correlation)")
    parser.add_argument("--eve-json", type=Path, default=None, help="Path to eve.json file (for Suricata alert correlation)")

    args = parser.parse_args()

    evaluate_models(
        args.predictions, 
        args.X_scaled, 
        args.preprocessor, 
        args.models_dir, 
        args.output_dir,
        original_flows_path=args.original_flows,
        eve_json_path=args.eve_json,
    )


if __name__ == "__main__":
    main()

