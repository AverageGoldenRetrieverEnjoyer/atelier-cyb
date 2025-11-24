#!/usr/bin/env python3
"""
Train unsupervised ML models: Isolation Forest, K-Means, and PyTorch Autoencoder.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """PyTorch Autoencoder for anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = None, hidden_dims: list = None):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            # Default: 3-4 layers with bottleneck
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        if latent_dim is None:
            latent_dim = input_dim // 8  # ~50% reduction in bottleneck
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (reverse structure)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_isolation_forest(X: np.ndarray, contamination: float = 0.1, n_estimators: int = 200) -> IsolationForest:
    """Train Isolation Forest model."""
    LOGGER.info("=" * 60)
    LOGGER.info("Training Isolation Forest")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Contamination: {contamination}")
    LOGGER.info(f"N estimators: {n_estimators}")
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X)
    
    # Get predictions and scores
    predictions = model.predict(X)
    scores = model.score_samples(X)
    
    anomaly_count = (predictions == -1).sum()
    LOGGER.info(f"Detected {anomaly_count:,} anomalies ({anomaly_count/len(X)*100:.2f}%)")
    LOGGER.info(f"Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    return model


def train_kmeans(X: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> Tuple[KMeans, int]:
    """Train K-Means with optimal k selection."""
    LOGGER.info("=" * 60)
    LOGGER.info("Training K-Means")
    LOGGER.info("=" * 60)
    
    k_min, k_max = k_range
    best_k = k_min
    best_score = -1
    best_model = None
    
    # Test different k values
    LOGGER.info(f"Testing k values from {k_min} to {k_max}...")
    
    # Use sample for faster evaluation
    sample_size = min(50000, len(X))
    X_sample = X[np.random.choice(len(X), sample_size, replace=False)] if len(X) > sample_size else X
    
    for k in range(k_min, k_max + 1):
        LOGGER.info(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_sample)
        
        # Use silhouette score to evaluate
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters
            score = silhouette_score(X_sample, labels)
            LOGGER.info(f"    Silhouette score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans
        else:
            LOGGER.warning(f"    Only 1 cluster found, skipping")
    
    LOGGER.info(f"Best k: {best_k} (silhouette score: {best_score:.4f})")
    
    # Train final model on full dataset
    LOGGER.info("Training final model on full dataset...")
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    final_model.fit(X)
    
    return final_model, best_k


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = None,
    hidden_dims: list = None,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = None,
) -> Tuple[Autoencoder, Dict]:
    """Train PyTorch Autoencoder."""
    LOGGER.info("=" * 60)
    LOGGER.info("Training Autoencoder")
    LOGGER.info("=" * 60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    
    input_dim = X.shape[1]
    
    # Create model
    model = Autoencoder(input_dim, latent_dim, hidden_dims)
    model = model.to(device)
    
    # Prepare data
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Training loop
    model.train()
    train_losses = []
    best_loss = float("inf")
    patience_counter = 0
    max_patience = 10
    
    LOGGER.info(f"Training for {epochs} epochs...")
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            LOGGER.info(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            LOGGER.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    LOGGER.info(f"Training complete. Final loss: {train_losses[-1]:.6f}")
    
    # Calculate reconstruction errors for threshold selection
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid memory issues
        reconstruction_errors_list = []
        batch_size_eval = min(batch_size * 4, 10000)  # Larger batches for eval
        
        for i in range(0, len(X), batch_size_eval):
            X_batch = torch.FloatTensor(X[i:i+batch_size_eval]).to(device)
            reconstructed_batch = model(X_batch)
            errors_batch = torch.mean((X_batch - reconstructed_batch) ** 2, dim=1).cpu().numpy()
            reconstruction_errors_list.append(errors_batch)
        
        reconstruction_errors = np.concatenate(reconstruction_errors_list)
    
    # Select threshold (95th percentile)
    threshold = np.percentile(reconstruction_errors, 95)
    LOGGER.info(f"Reconstruction error threshold (95th percentile): {threshold:.6f}")
    
    training_info = {
        "train_losses": train_losses,
        "threshold": threshold,
        "reconstruction_errors": reconstruction_errors,
    }
    
    return model, training_info


def train_all_models(
    X_path: Path,
    output_dir: Path,
    contamination: float = 0.1,
    k_range: Tuple[int, int] = (2, 10),
    autoencoder_epochs: int = 50,
    optimized_params_path: Path = None,
) -> Dict:
    """Train all three models, optionally using optimized hyperparameters."""
    
    # Load preprocessed data
    LOGGER.info(f"Loading preprocessed data from {X_path}")
    X = pd.read_csv(X_path).values
    LOGGER.info(f"Data shape: {X.shape}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load optimized parameters if provided
    optimized_params = None
    if optimized_params_path and optimized_params_path.exists():
        LOGGER.info(f"Loading optimized hyperparameters from {optimized_params_path}")
        try:
            optimized_params = joblib.load(optimized_params_path)
            LOGGER.info("Using optimized hyperparameters")
        except Exception as e:
            LOGGER.warning(f"Failed to load optimized parameters: {e}. Using defaults.")
            optimized_params = None
    
    # Train Isolation Forest
    if optimized_params and optimized_params.get("isolation_forest") and optimized_params["isolation_forest"] is not None:
        if_params = optimized_params["isolation_forest"]["best_params"]
        LOGGER.info(f"Using optimized IF parameters: {if_params}")
        if_model = IsolationForest(
            contamination=if_params["contamination"],
            n_estimators=if_params["n_estimators"],
            max_samples=if_params["max_samples"],
            max_features=if_params["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        if_model.fit(X)
    else:
        if_model = train_isolation_forest(X, contamination)
    joblib.dump(if_model, output_dir / "isolation_forest.joblib")
    
    # Get IF predictions
    if_predictions = if_model.predict(X)
    if_scores = if_model.score_samples(X)
    
    # Train K-Means
    if optimized_params and optimized_params.get("kmeans") and optimized_params["kmeans"] is not None:
        kmeans_params = optimized_params["kmeans"]["best_params"]
        LOGGER.info(f"Using optimized K-Means parameters: {kmeans_params}")
        kmeans_model = KMeans(
            n_clusters=kmeans_params["n_clusters"],
            n_init=kmeans_params["n_init"],
            max_iter=kmeans_params["max_iter"],
            algorithm=kmeans_params["algorithm"],
            random_state=42,
            n_jobs=-1,
        )
        kmeans_model.fit(X)
        best_k = kmeans_params["n_clusters"]
    else:
        kmeans_model, best_k = train_kmeans(X, k_range)
    joblib.dump(kmeans_model, output_dir / "kmeans.joblib")
    
    # Get K-Means predictions
    kmeans_labels = kmeans_model.predict(X)
    
    # Train Autoencoder
    if optimized_params and optimized_params.get("autoencoder") and optimized_params["autoencoder"] is not None:
        ae_params = optimized_params["autoencoder"]["best_params"]
        LOGGER.info(f"Using optimized Autoencoder parameters")
        input_dim = X.shape[1]
        latent_dim = max(8, int(input_dim * ae_params["latent_dim_ratio"]))
        hidden_dim1 = max(16, int(input_dim * ae_params["hidden_dim1_ratio"]))
        hidden_dim2 = max(8, int(input_dim * ae_params["hidden_dim2_ratio"]))
        hidden_dims = [hidden_dim1, hidden_dim2]
        
        ae_model, ae_info = train_autoencoder(
            X,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            epochs=ae_params["epochs"],
            batch_size=ae_params["batch_size"],
            learning_rate=ae_params["learning_rate"],
        )
    else:
        ae_model, ae_info = train_autoencoder(X, epochs=autoencoder_epochs)
    torch.save(ae_model.state_dict(), output_dir / "autoencoder.pth")
    joblib.dump(ae_info, output_dir / "autoencoder_info.joblib")
    
    # Get Autoencoder predictions (based on threshold)
    ae_predictions = (ae_info["reconstruction_errors"] > ae_info["threshold"]).astype(int)
    ae_predictions = np.where(ae_predictions == 1, -1, 1)  # Convert to IF format (-1 = anomaly)
    
    # Save all predictions
    predictions_df = pd.DataFrame({
        "isolation_forest": if_predictions,
        "isolation_forest_score": if_scores,
        "kmeans_cluster": kmeans_labels,
        "autoencoder": ae_predictions,
        "autoencoder_reconstruction_error": ae_info["reconstruction_errors"],
    })
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    LOGGER.info("=" * 60)
    LOGGER.info("Training Summary")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Isolation Forest: {np.sum(if_predictions == -1):,} anomalies")
    LOGGER.info(f"K-Means: {best_k} clusters")
    LOGGER.info(f"Autoencoder: {np.sum(ae_predictions == -1):,} anomalies")
    LOGGER.info(f"Models saved to: {output_dir}")
    
    return {
        "isolation_forest": if_model,
        "kmeans": kmeans_model,
        "autoencoder": ae_model,
        "predictions": predictions_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train unsupervised ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--X-scaled", type=Path, required=True, help="Path to preprocessed X_scaled.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for models")
    parser.add_argument("--contamination", type=float, default=0.1, help="Isolation Forest contamination")
    parser.add_argument("--k-min", type=int, default=2, help="K-Means minimum k")
    parser.add_argument("--k-max", type=int, default=10, help="K-Means maximum k")
    parser.add_argument("--ae-epochs", type=int, default=50, help="Autoencoder training epochs")
    parser.add_argument("--optimized-params", type=Path, default=None, help="Path to optimized hyperparameters file")

    args = parser.parse_args()

    train_all_models(
        args.X_scaled,
        args.output_dir,
        args.contamination,
        (args.k_min, args.k_max),
        args.ae_epochs,
        optimized_params_path=args.optimized_params,
    )


if __name__ == "__main__":
    main()

