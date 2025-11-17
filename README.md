# Memory-Efficient Suricata Flow Anomaly Detection

This repo contains a two-stage workflow to train and evaluate a flow-based anomaly model on Suricata `eve.json` data while keeping results correlated with Suricata alerts for validation.

## Repository layout

```
src/                # Main application code
scripts/            # Utility scripts (export, preprocessing, decompression, etc.)
legacy/             # Archived experimental pipelines
data/               # Suricata eve.json and PCAP archives (not tracked in git)
outputs/            # Generated artifacts (CSV, plots, models)
```

## Requirements

- Python 3.10+
- pip packages listed in `requirements.txt`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Main pipeline

`src/anomaly_pipeline.py` loads Suricata flows, engineers features, trains an Isolation Forest, correlates predictions with alert events, and exports both CSV summaries and presentation-ready plots.

Key outputs (per run):

- `flows_with_anomalies.csv`: All flows with engineered features, anomaly labels, and alert metadata.
- `summary.json`: Global metrics (flows processed, anomalies, overlap with alerts).
- `plots/*.png`: Dataset & training visualizations (volume distribution, anomaly scores, alert overlap).
- Optional `models/*.joblib` when `--save-model` is provided.

### Usage

```bash
python3 src/anomaly_pipeline.py \
  --eve data/eve.json \
  --output-dir outputs/full \
  --progress \
  --save-model outputs/models/flow_iforest.joblib
```

Useful flags:

- `--max-flows N`: limit number of flow events (great for demos).
- `--contamination 0.01`: expected anomaly ratio passed to IsolationForest.
- `--no-plots`: skip matplotlib/seaborn rendering.
- `--progress`: show tqdm parsing progress.

## Makefile helpers

The Makefile provides two targets:

- `make demo`: runs the pipeline on `data/eve_sample.json` with modest limits, producing artifacts under `outputs/demo/`.
- `make full`: executes the pipeline on the full `data/eve.json`, saving plots, CSVs, and the trained model under `outputs/full/`.

Both targets assume the required files already exist under `data/`.

## Visualization outputs

The pipeline automatically generates several PNG charts to include in presentations:

1. **bytes_distribution** – histogram of per-flow volume (log count).
2. **bytes_scatter** – bytes-to-server vs bytes-to-client scatter (log-log) colored by anomaly label.
3. **anomaly_scores** – distribution of IsolationForest scores.
4. **alert_overlap** – bar chart showing how many anomalous flows also triggered Suricata alerts.

All plots are saved under `outputs/<run>/plots/`.

## Repo housekeeping

- Utility scripts live in `scripts/` to keep the main source clean.
- Archived experiments sit in `legacy/` for reference.
- `.gitignore` prevents large data blobs, databases, and generated artifacts from polluting the git history.

## Next steps

1. Run `make demo` to validate everything on a small subset.
2. Run `make full` (preferably on a beefier machine) to process the complete dataset.
3. Use `outputs/.../summary.json` and the PNG graphs in presentations to justify model quality by comparing anomalies vs alerts.


