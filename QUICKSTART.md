# Quick Start Guide – Flow Anomaly Pipeline

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data layout

```
data/
  eve_sample.json     # lightweight subset for demos
  eve.json            # full Suricata export (not tracked in git)
```

Ensure the `data/` directory exists at the project root before running the pipeline.

## Quick demo

```bash
make demo
```

This runs `src/anomaly_pipeline.py` on `data/eve_sample.json`, generating results under `outputs/demo/`.

## Full dataset

```bash
make full
```

Artifacts (CSV, plots, model) are stored under `outputs/full/`.

## What the pipeline does

1. Loads Suricata `flow` records from `eve.json` (optionally capped via `--max-flows`).
2. Engineers statistical features (volume, ratios, duration, protocol flags).
3. Trains an IsolationForest to flag anomalous flows.
4. Correlates anomalies with Suricata `alert` events to gauge alignment.
5. Exports CSV summaries plus presentation-ready PNG charts.

## CLI reference

```bash
python3 src/anomaly_pipeline.py \
  --eve data/eve.json \
  --output-dir outputs/full \
  --progress \
  --save-model outputs/models/flow_iforest.joblib
```

Useful flags:

- `--max-flows 200000` – sample for faster experiments.
- `--contamination 0.01` – expected anomaly fraction.
- `--no-plots` – skip charts when running headless.
- `--progress` – show tqdm parsing progress.

## Generated artifacts

- `flows_with_anomalies.csv` – enriched flows + anomaly labels + alert metadata.
- `summary.json` – key metrics (flows processed, anomalies, overlap with alerts).
- `plots/*.png` – bytes distribution, scatter, anomaly score histogram, alert overlap.
- `models/*.joblib` – persisted IsolationForest model (only when `--save-model` is used).

## Troubleshooting

- **Parsing feels slow** → run with `--max-flows` for exploratory runs.
- **Matplotlib errors** → install system fonts or use `--no-plots`.
- **Memory pressure** → split `eve.json` or run on a machine with more RAM; IForest loads all features.

For more details, see `README.md`.

