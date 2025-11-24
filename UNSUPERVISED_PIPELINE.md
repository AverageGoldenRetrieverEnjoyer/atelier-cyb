# Unsupervised Intrusion Detection Pipeline

Complete pipeline for training unsupervised ML models (Isolation Forest, K-Means, Autoencoder) on network flow data extracted from PCAP files.

## Quick Start

```bash
# Full pipeline (recommended)
make unsupervised-pipeline

# Or step by step:
make extract-flows          # Extract flows from PCAPs
make preprocess-flows        # Preprocess and scale
make train-unsupervised     # Train all models
make evaluate-unsupervised  # Evaluate and visualize
```

## Pipeline Steps

### 1. Flow Extraction (`src/pcap_to_flows.py`)
- Extracts flow-level features from PCAP files using `nfstream`
- Features: packet/byte counts, duration, protocol, IPs, ports, inter-arrival times, TCP flags, etc.
- Outputs: `attack_flows.csv`, `benign_flows.csv`

### 2. Preprocessing (`src/preprocess_flows.py`)
- Combines attack and benign flows (removes labels for unsupervised)
- Handles missing values (drops >50% missing, imputes others)
- Encodes categorical features (hash for high cardinality, one-hot for low)
- Tests and selects best scaler (StandardScaler, RobustScaler, MinMaxScaler)
- Outputs: `X_scaled.csv`, `preprocessor.joblib`

### 3. Model Training (`src/train_unsupervised.py`)
- **Isolation Forest**: Detects anomalies (contamination=0.1 by default)
- **K-Means**: Clustering (tests k=2-10, selects best via silhouette score)
- **PyTorch Autoencoder**: Deep learning anomaly detection (reconstruction error)
- Outputs: `isolation_forest.joblib`, `kmeans.joblib`, `autoencoder.pth`, `predictions.csv`

### 4. Evaluation (`src/evaluate_models.py`)
- Isolation Forest: Score distributions, anomaly analysis
- K-Means: PCA/t-SNE visualizations, cluster characteristics
- Autoencoder: Reconstruction error analysis, threshold selection
- Model Agreement: Comparison between models
- Outputs: Multiple PNG plots, `evaluation.json`

## Usage Examples

### Basic Usage
```bash
python src/run_unsupervised_pipeline.py \
    --attack-pcap data/attack_data.pcap \
    --benign-pcap data/benign_data.pcap
```

### With Downsampling (for large datasets)
```bash
python src/run_unsupervised_pipeline.py \
    --attack-pcap data/attack_data.pcap \
    --benign-pcap data/benign_data.pcap \
    --sample-rate 0.2  # Use 20% of data
```

### Custom Model Parameters
```bash
python src/run_unsupervised_pipeline.py \
    --attack-pcap data/attack_data.pcap \
    --benign-pcap data/benign_data.pcap \
    --contamination 0.15 \
    --k-min 3 \
    --k-max 8 \
    --ae-epochs 100
```

### Step-by-Step Execution
```bash
# Extract flows
python src/pcap_to_flows.py \
    --attack-pcap data/attack_data.pcap \
    --benign-pcap data/benign_data.pcap \
    --output-dir outputs/flows \
    --sample-rate 0.2

# Preprocess
python src/preprocess_flows.py \
    --attack-flows outputs/flows/attack_flows.csv \
    --benign-flows outputs/flows/benign_flows.csv \
    --output-dir outputs/preprocessed

# Train
python src/train_unsupervised.py \
    --X-scaled outputs/preprocessed/X_scaled.csv \
    --output-dir outputs/models \
    --contamination 0.1

# Evaluate
python src/evaluate_models.py \
    --predictions outputs/models/predictions.csv \
    --X-scaled outputs/preprocessed/X_scaled.csv \
    --output-dir outputs/plots
```

## Output Structure

Each run creates a timestamped directory:
```
outputs/
  └── YYYYMMDD_HHMMSS/
      ├── flows/
      │   ├── attack_flows.csv
      │   └── benign_flows.csv
      ├── preprocessed/
      │   ├── X_scaled.csv
      │   └── preprocessor.joblib
      ├── models/
      │   ├── isolation_forest.joblib
      │   ├── kmeans.joblib
      │   ├── autoencoder.pth
      │   ├── autoencoder_info.joblib
      │   └── predictions.csv
      ├── results/
      │   └── evaluation.json
      └── plots/
          ├── isolation_forest_analysis.png
          ├── kmeans_analysis.png
          ├── autoencoder_analysis.png
          └── model_agreement.png
```

## Memory Efficiency

- **Downsampling**: Use `--sample-rate` to process a fraction of data
- **Chunked Processing**: Flow extraction processes in chunks (10k flows at a time)
- **Incremental Evaluation**: Autoencoder evaluation uses batched inference
- **Streaming**: PCAP processing is memory-efficient with nfstream

## Model Details

### Isolation Forest
- Unsupervised anomaly detection
- Contamination: Expected fraction of anomalies (default: 0.1 = 10%)
- Output: Anomaly scores (lower = more anomalous)

### K-Means
- Clustering to find patterns
- Tests k=2-10, selects best via silhouette score
- Output: Cluster assignments

### Autoencoder
- Deep learning anomaly detection
- Architecture: Encoder → Latent (bottleneck) → Decoder
- Loss: MSE reconstruction error
- Threshold: 95th percentile of reconstruction errors
- Output: Reconstruction errors (higher = more anomalous)

## Requirements

See `requirements.txt`. Key dependencies:
- `nfstream` - Flow extraction
- `torch` - Autoencoder
- `scikit-learn` - Isolation Forest, K-Means, preprocessing
- `matplotlib`, `seaborn` - Visualization

## Troubleshooting

**Out of memory?**
- Use `--sample-rate 0.2` to downsample
- Use `--max-flows` to limit flow extraction
- Process in smaller batches

**Slow processing?**
- Downsample data first
- Reduce `--ae-epochs` for faster autoencoder training
- Skip t-SNE visualization (auto-skipped for >5000 samples)

**Models not detecting anomalies?**
- Adjust `--contamination` for Isolation Forest
- Check data quality (ensure attack traffic is present)
- Review preprocessing (missing values, scaling)

