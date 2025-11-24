#!/usr/bin/env python3
"""
Hyperparameter optimization for unsupervised ML models using Optuna.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
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

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list, dropout: float = 0.1):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
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
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def optimize_isolation_forest(
    X: np.ndarray,
    n_trials: int = 20,
    timeout: int = None,
    n_jobs: int = 1,
) -> Dict:
    """
    Optimize Isolation Forest hyperparameters using Optuna.
    
    Args:
        X: Training data
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary with best parameters and model
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Optimizing Isolation Forest Hyperparameters")
    LOGGER.info("=" * 60)
    
    # Use a sample for faster optimization
    sample_size = min(50000, len(X))
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    def objective(trial):
        contamination = trial.suggest_float("contamination", 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
        max_samples = trial.suggest_categorical("max_samples", ["auto", 0.5, 0.7, 0.9])
        max_features = trial.suggest_float("max_features", 0.5, 1.0)
        
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        
        model.fit(X_sample)
        predictions = model.predict(X_sample)
        scores = model.score_samples(X_sample)
        
        # Use negative silhouette score as metric (lower is better for minimization)
        # Convert predictions to binary labels for silhouette score
        if len(np.unique(predictions)) > 1:
            # Use silhouette score on the score_samples (lower scores = anomalies)
            # Create binary labels based on median score
            median_score = np.median(scores)
            binary_labels = (scores < median_score).astype(int)
            
            if len(np.unique(binary_labels)) > 1:
                silhouette = silhouette_score(X_sample, binary_labels)
                # We want to maximize silhouette, so return negative for minimization
                return -silhouette
            else:
                return 1.0  # Penalty if all same label
        else:
            return 1.0  # Penalty if all same prediction
    
    study = optuna.create_study(direction="minimize", study_name="isolation_forest")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=True)
    
    best_params = study.best_params
    LOGGER.info(f"Best parameters: {best_params}")
    LOGGER.info(f"Best score: {-study.best_value:.4f}")
    
    # Train final model with best parameters on full dataset
    final_model = IsolationForest(
        contamination=best_params["contamination"],
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        max_features=best_params["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X)
    
    return {
        "model": final_model,
        "best_params": best_params,
        "best_score": -study.best_value,
        "study": study,
    }


def optimize_kmeans(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 20),
    n_trials: int = 20,
    timeout: int = None,
    n_jobs: int = 1,
) -> Dict:
    """
    Optimize K-Means hyperparameters using Optuna.
    
    Args:
        X: Training data
        k_range: Range of k values to test
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary with best parameters and model
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Optimizing K-Means Hyperparameters")
    LOGGER.info("=" * 60)
    
    # Use a sample for faster optimization
    sample_size = min(50000, len(X))
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    k_min, k_max = k_range
    
    def objective(trial):
        n_clusters = trial.suggest_int("n_clusters", k_min, k_max)
        n_init = trial.suggest_int("n_init", 5, 20)
        max_iter = trial.suggest_int("max_iter", 100, 500, step=50)
        algorithm = trial.suggest_categorical("algorithm", ["lloyd", "elkan"])
        
        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            algorithm=algorithm,
            random_state=42,
            n_jobs=-1,
        )
        
        labels = model.fit_predict(X_sample)
        
        # Use silhouette score as metric
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X_sample, labels)
            # We want to maximize silhouette, so return negative for minimization
            return -silhouette
        else:
            return 1.0  # Penalty if all same cluster
    
    study = optuna.create_study(direction="minimize", study_name="kmeans")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=True)
    
    best_params = study.best_params
    LOGGER.info(f"Best parameters: {best_params}")
    LOGGER.info(f"Best score: {-study.best_value:.4f}")
    
    # Train final model with best parameters on full dataset
    final_model = KMeans(
        n_clusters=best_params["n_clusters"],
        n_init=best_params["n_init"],
        max_iter=best_params["max_iter"],
        algorithm=best_params["algorithm"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X)
    
    return {
        "model": final_model,
        "best_params": best_params,
        "best_score": -study.best_value,
        "study": study,
    }


def optimize_autoencoder(
    X: np.ndarray,
    n_trials: int = 20,
    timeout: int = None,
    n_jobs: int = 1,
    device: str = None,
    max_epochs: int = 50,
) -> Dict:
    """
    Optimize Autoencoder hyperparameters using Optuna.
    
    Args:
        X: Training data
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        n_jobs: Number of parallel jobs
        device: Device to use for training
        max_epochs: Maximum number of epochs per trial
    
    Returns:
        Dictionary with best parameters and model
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Optimizing Autoencoder Hyperparameters")
    LOGGER.info("=" * 60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    
    input_dim = X.shape[1]
    
    # Use a sample for faster optimization
    sample_size = min(20000, len(X))
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    def objective(trial):
        # Suggest hyperparameters
        latent_dim_ratio = trial.suggest_float("latent_dim_ratio", 0.05, 0.3)
        hidden_dim1_ratio = trial.suggest_float("hidden_dim1_ratio", 0.3, 0.7)
        hidden_dim2_ratio = trial.suggest_float("hidden_dim2_ratio", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        epochs = trial.suggest_int("epochs", 10, max_epochs, step=10)
        
        # Calculate dimensions
        latent_dim = max(8, int(input_dim * latent_dim_ratio))
        hidden_dim1 = max(16, int(input_dim * hidden_dim1_ratio))
        hidden_dim2 = max(8, int(input_dim * hidden_dim2_ratio))
        hidden_dims = [hidden_dim1, hidden_dim2]
        
        # Create model
        model = Autoencoder(input_dim, latent_dim, hidden_dims, dropout)
        model = model.to(device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_sample)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop (simplified for optimization)
        model.train()
        best_loss = float("inf")
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(epochs):
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
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                break
            
            # Report intermediate value for pruning
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate final reconstruction error
        model.eval()
        with torch.no_grad():
            X_tensor_full = torch.FloatTensor(X_sample).to(device)
            reconstructed = model(X_tensor_full)
            final_loss = criterion(reconstructed, X_tensor_full).item()
        
        return final_loss
    
    study = optuna.create_study(direction="minimize", study_name="autoencoder")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=True)
    
    best_params = study.best_params
    LOGGER.info(f"Best parameters: {best_params}")
    LOGGER.info(f"Best score: {study.best_value:.6f}")
    
    # Calculate final dimensions from best parameters
    latent_dim = max(8, int(input_dim * best_params["latent_dim_ratio"]))
    hidden_dim1 = max(16, int(input_dim * best_params["hidden_dim1_ratio"]))
    hidden_dim2 = max(8, int(input_dim * best_params["hidden_dim2_ratio"]))
    hidden_dims = [hidden_dim1, hidden_dim2]
    
    # Train final model with best parameters on full dataset
    final_model = Autoencoder(
        input_dim,
        latent_dim,
        hidden_dims,
        best_params["dropout"]
    )
    final_model = final_model.to(device)
    
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=best_params["batch_size"],
        shuffle=True
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    final_model.train()
    train_losses = []
    best_loss = float("inf")
    patience_counter = 0
    max_patience = 10
    
    LOGGER.info(f"Training final model for {best_params['epochs']} epochs...")
    for epoch in tqdm(range(best_params["epochs"]), desc="Final training"):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = final_model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            LOGGER.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Calculate reconstruction errors for threshold selection
    final_model.eval()
    with torch.no_grad():
        reconstruction_errors_list = []
        batch_size_eval = min(best_params["batch_size"] * 4, 10000)
        
        for i in range(0, len(X), batch_size_eval):
            X_batch = torch.FloatTensor(X[i:i+batch_size_eval]).to(device)
            reconstructed_batch = final_model(X_batch)
            errors_batch = torch.mean((X_batch - reconstructed_batch) ** 2, dim=1).cpu().numpy()
            reconstruction_errors_list.append(errors_batch)
        
        reconstruction_errors = np.concatenate(reconstruction_errors_list)
    
    threshold = np.percentile(reconstruction_errors, 95)
    
    training_info = {
        "train_losses": train_losses,
        "threshold": threshold,
        "reconstruction_errors": reconstruction_errors,
        "best_params": best_params,
    }
    
    return {
        "model": final_model,
        "best_params": best_params,
        "best_score": study.best_value,
        "study": study,
        "training_info": training_info,
    }


def optimize_all_models(
    X_path: Path,
    output_dir: Path,
    n_trials: int = 20,
    timeout: int = None,
    optimize_if: bool = True,
    optimize_kmeans: bool = True,
    optimize_ae: bool = True,
    k_range: Tuple[int, int] = (2, 20),
) -> Dict:
    """
    Optimize hyperparameters for all models.
    
    Args:
        X_path: Path to preprocessed data
        output_dir: Output directory for optimized models
        n_trials: Number of trials per model
        timeout: Maximum time in seconds per model
        optimize_if: Whether to optimize Isolation Forest
        optimize_kmeans: Whether to optimize K-Means
        optimize_ae: Whether to optimize Autoencoder
        k_range: Range of k values for K-Means
    
    Returns:
        Dictionary with optimized models and parameters
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Hyperparameter Optimization")
    LOGGER.info("=" * 60)
    
    # Load preprocessed data
    LOGGER.info(f"Loading preprocessed data from {X_path}")
    X = pd.read_csv(X_path).values
    LOGGER.info(f"Data shape: {X.shape}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    # Optimize Isolation Forest
    if optimize_if:
        if_result = optimize_isolation_forest(X, n_trials=n_trials, timeout=timeout)
        joblib.dump(if_result["model"], output_dir / "isolation_forest_optimized.joblib")
        joblib.dump(if_result["best_params"], output_dir / "isolation_forest_params.joblib")
        results["isolation_forest"] = if_result
        LOGGER.info(f"Isolation Forest optimization complete. Best score: {if_result['best_score']:.4f}")
    
    # Optimize K-Means
    if optimize_kmeans:
        kmeans_result = optimize_kmeans(X, k_range=k_range, n_trials=n_trials, timeout=timeout)
        joblib.dump(kmeans_result["model"], output_dir / "kmeans_optimized.joblib")
        joblib.dump(kmeans_result["best_params"], output_dir / "kmeans_params.joblib")
        results["kmeans"] = kmeans_result
        LOGGER.info(f"K-Means optimization complete. Best score: {kmeans_result['best_score']:.4f}")
    
    # Optimize Autoencoder
    if optimize_ae:
        ae_result = optimize_autoencoder(X, n_trials=n_trials, timeout=timeout)
        torch.save(ae_result["model"].state_dict(), output_dir / "autoencoder_optimized.pth")
        joblib.dump(ae_result["training_info"], output_dir / "autoencoder_info_optimized.joblib")
        joblib.dump(ae_result["best_params"], output_dir / "autoencoder_params.joblib")
        results["autoencoder"] = ae_result
        LOGGER.info(f"Autoencoder optimization complete. Best score: {ae_result['best_score']:.6f}")
    
    # Save optimization summary
    summary = {
        "isolation_forest": {
            "best_params": results.get("isolation_forest", {}).get("best_params"),
            "best_score": results.get("isolation_forest", {}).get("best_score"),
        } if optimize_if else None,
        "kmeans": {
            "best_params": results.get("kmeans", {}).get("best_params"),
            "best_score": results.get("kmeans", {}).get("best_score"),
        } if optimize_kmeans else None,
        "autoencoder": {
            "best_params": results.get("autoencoder", {}).get("best_params"),
            "best_score": results.get("autoencoder", {}).get("best_score"),
        } if optimize_ae else None,
    }
    
    joblib.dump(summary, output_dir / "optimization_summary.joblib")
    
    LOGGER.info("=" * 60)
    LOGGER.info("Optimization Complete!")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Optimized models saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for unsupervised ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--X-scaled", type=Path, required=True, help="Path to preprocessed X_scaled.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for optimized models")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials per model")
    parser.add_argument("--timeout", type=int, default=None, help="Maximum time in seconds per model")
    parser.add_argument("--k-min", type=int, default=2, help="K-Means minimum k")
    parser.add_argument("--k-max", type=int, default=20, help="K-Means maximum k")
    parser.add_argument("--skip-if", action="store_true", help="Skip Isolation Forest optimization")
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip K-Means optimization")
    parser.add_argument("--skip-ae", action="store_true", help="Skip Autoencoder optimization")
    
    args = parser.parse_args()
    
    optimize_all_models(
        args.X_scaled,
        args.output_dir,
        n_trials=args.n_trials,
        timeout=args.timeout,
        optimize_if=not args.skip_if,
        optimize_kmeans=not args.skip_kmeans,
        optimize_ae=not args.skip_ae,
        k_range=(args.k_min, args.k_max),
    )

