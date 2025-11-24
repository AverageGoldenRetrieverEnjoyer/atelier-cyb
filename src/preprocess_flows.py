#!/usr/bin/env python3
"""
Preprocess flow data for unsupervised ML training.
Handles missing values, feature engineering, scaling, and downsampling.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


def load_and_combine_flows(attack_path: Path, benign_path: Path) -> pd.DataFrame:
    """Load attack and benign flows, combine without labels."""
    LOGGER.info("Loading flow data...")
    
    attack_df = pd.read_csv(attack_path)
    benign_df = pd.read_csv(benign_path)
    
    LOGGER.info(f"Attack flows: {len(attack_df):,}")
    LOGGER.info(f"Benign flows: {len(benign_df):,}")
    
    # Combine without labels (unsupervised learning)
    combined = pd.concat([attack_df, benign_df], ignore_index=True)
    LOGGER.info(f"Combined flows: {len(combined):,}")
    
    return combined


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values based on context."""
    LOGGER.info("Handling missing values...")
    
    initial_cols = len(df.columns)
    initial_rows = len(df)
    
    # Drop columns with >50% missing values
    missing_threshold = 0.5
    cols_to_drop = []
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct > missing_threshold:
            cols_to_drop.append(col)
            LOGGER.info(f"  Dropping {col}: {missing_pct*100:.1f}% missing")
    
    df = df.drop(columns=cols_to_drop)
    
    # For remaining missing values:
    # - Numeric: impute with median
    # - Categorical: impute with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            LOGGER.debug(f"  Imputed {col} (numeric) with median: {median_val}")
    
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
            df[col].fillna(mode_val, inplace=True)
            LOGGER.debug(f"  Imputed {col} (categorical) with mode: {mode_val}")
    
    LOGGER.info(f"  Dropped {initial_cols - len(df.columns)} columns")
    LOGGER.info(f"  Final shape: {df.shape}")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Encode categorical features."""
    LOGGER.info("Encoding categorical features...")
    
    encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    for col in categorical_cols:
        # For high cardinality (like IPs), use hash encoding
        # For low cardinality, use one-hot encoding
        unique_count = df[col].nunique()
        
        if unique_count > 100:  # High cardinality (IPs, etc.)
            # Hash encoding to fixed size
            df[f"{col}_hash"] = df[col].apply(lambda x: hash(str(x)) % 1000)
            df = df.drop(columns=[col])
            encoders[col] = "hash"
            LOGGER.info(f"  {col}: hash encoding ({unique_count} unique values)")
        else:  # Low cardinality (protocol, app name, etc.)
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoders[col] = "onehot"
            LOGGER.info(f"  {col}: one-hot encoding ({unique_count} unique values)")
    
    return df, encoders


def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove constant features (zero variance)."""
    LOGGER.info("Removing constant features...")
    
    initial_cols = len(df.columns)
    
    # Remove constant columns
    constant_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        df = df.drop(columns=constant_cols)
        LOGGER.info(f"  Removed {len(constant_cols)} constant features")
    
    LOGGER.info(f"  Features: {initial_cols} -> {len(df.columns)}")
    
    return df


def select_best_scaler(X: pd.DataFrame) -> Tuple[object, str]:
    """Test different scalers and select the best one."""
    LOGGER.info("Testing scalers...")
    
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
    }
    
    # Use a sample for testing (faster)
    sample_size = min(10000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X
    
    best_scaler = None
    best_name = "standard"
    best_score = float("inf")
    
    for name, scaler in scalers.items():
        try:
            X_scaled = scaler.fit_transform(X_sample)
            
            # Evaluate based on:
            # 1. No infinite values
            # 2. Reasonable range
            # 3. Low variance in scaling (stability)
            
            has_inf = np.isinf(X_scaled).any()
            has_nan = np.isnan(X_scaled).any()
            
            if has_inf or has_nan:
                LOGGER.warning(f"  {name}: Contains inf/nan values")
                continue
            
            # Check range (should be reasonable)
            min_val = X_scaled.min()
            max_val = X_scaled.max()
            
            # Score: prefer scalers with reasonable ranges
            if name == "minmax":
                score = abs(max_val - 1.0) + abs(min_val - 0.0)  # Should be [0,1]
            else:
                # For standard/robust, prefer smaller absolute ranges
                score = abs(max_val) + abs(min_val)
            
            LOGGER.info(f"  {name}: range [{min_val:.2f}, {max_val:.2f}], score: {score:.2f}")
            
            if score < best_score:
                best_score = score
                best_scaler = scaler
                best_name = name
        except Exception as e:
            LOGGER.warning(f"  {name}: Error during testing - {e}")
            continue
    
    LOGGER.info(f"Selected scaler: {best_name}")
    return best_scaler, best_name


def preprocess_flows(
    attack_path: Path,
    benign_path: Path,
    output_dir: Path,
    sample_rate: float = None,
) -> Tuple[pd.DataFrame, object, Dict]:
    """Main preprocessing pipeline."""
    
    # Load and combine
    df = load_and_combine_flows(attack_path, benign_path)
    
    # Save original flows (before any preprocessing) for correlation with alerts
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep identifying columns that will be used for correlation
    identifying_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", 
                        "first_seen", "last_seen", "packets", "bytes"]
    # Only save columns that exist
    available_cols = [col for col in identifying_cols if col in df.columns]
    original_flows = df[available_cols].copy()
    original_flows.to_csv(output_dir / "original_flows.csv", index=False)
    LOGGER.info(f"Saved original flows metadata ({len(available_cols)} columns) to {output_dir / 'original_flows.csv'}")
    
    # Downsample if requested
    if sample_rate is not None and sample_rate < 1.0:
        original_size = len(df)
        df = df.sample(frac=sample_rate, random_state=42).reset_index(drop=True)
        LOGGER.info(f"Downsampled: {original_size:,} -> {len(df):,} flows ({sample_rate*100:.1f}%)")
        # Also downsample original_flows to match
        original_flows = original_flows.sample(frac=sample_rate, random_state=42).reset_index(drop=True)
        original_flows.to_csv(output_dir / "original_flows.csv", index=False)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Remove constant features
    df = remove_constant_features(df)
    
    # Separate features (all columns are now numeric)
    X = df.select_dtypes(include=[np.number])
    
    # Select and fit scaler
    scaler, scaler_name = select_best_scaler(X)
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    LOGGER.info(f"Final preprocessed data shape: {X_scaled_df.shape}")
    
    # Save preprocessed data
    output_dir.mkdir(parents=True, exist_ok=True)
    X_scaled_df.to_csv(output_dir / "X_scaled.csv", index=False)
    
    # Save preprocessor (scaler + encoders info)
    preprocessor = {
        "scaler": scaler,
        "scaler_name": scaler_name,
        "encoders": encoders,
        "feature_names": X.columns.tolist(),
    }
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    
    LOGGER.info(f"Saved preprocessed data to {output_dir}")
    
    return X_scaled_df, scaler, preprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess flow data for unsupervised ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attack-flows", type=Path, required=True, help="Path to attack flows CSV")
    parser.add_argument("--benign-flows", type=Path, required=True, help="Path to benign flows CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Downsampling rate (0.0-1.0). None = no downsampling",
    )

    args = parser.parse_args()

    preprocess_flows(
        args.attack_flows,
        args.benign_flows,
        args.output_dir,
        args.sample_rate,
    )


if __name__ == "__main__":
    main()

