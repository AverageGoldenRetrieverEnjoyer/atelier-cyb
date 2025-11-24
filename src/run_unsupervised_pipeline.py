#!/usr/bin/env python3
"""
Main orchestrator for unsupervised intrusion detection pipeline.
Runs: PCAP extraction → Preprocessing → Training → Evaluation
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
LOGGER = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and log the result."""
    LOGGER.info("=" * 60)
    LOGGER.info(description)
    LOGGER.info("=" * 60)
    LOGGER.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        LOGGER.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        LOGGER.error(f"✗ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete unsupervised intrusion detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input files
    parser.add_argument("--attack-pcap", type=Path, required=True, help="Path to attack PCAP file")
    parser.add_argument("--benign-pcap", type=Path, required=True, help="Path to benign PCAP file")
    
    # Output directory
    parser.add_argument("--output-base", type=Path, default=Path("outputs"), help="Base output directory")
    
    # Processing options
    parser.add_argument("--sample-rate", type=float, default=None, help="Downsampling rate (0.0-1.0)")
    parser.add_argument("--max-flows", type=int, default=None, help="Max flows per PCAP file")
    
    # Model parameters
    parser.add_argument("--contamination", type=float, default=0.1, help="Isolation Forest contamination")
    parser.add_argument("--k-min", type=int, default=2, help="K-Means minimum k")
    parser.add_argument("--k-max", type=int, default=10, help="K-Means maximum k")
    parser.add_argument("--ae-epochs", type=int, default=50, help="Autoencoder epochs")
    
    # Step selection
    parser.add_argument("--skip-extract", action="store_true", help="Skip flow extraction")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()
    
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_base / run_id
    
    LOGGER.info("=" * 60)
    LOGGER.info("Unsupervised Intrusion Detection Pipeline")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Run ID: {run_id}")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Attack PCAP: {args.attack_pcap}")
    LOGGER.info(f"Benign PCAP: {args.benign_pcap}")
    
    # Create directory structure
    flows_dir = output_dir / "flows"
    preprocessed_dir = output_dir / "preprocessed"
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    
    for d in [flows_dir, preprocessed_dir, models_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Step 1: Extract flows
    if not args.skip_extract:
        cmd = [
            sys.executable,
            "src/pcap_to_flows.py",
            "--attack-pcap", str(args.attack_pcap),
            "--benign-pcap", str(args.benign_pcap),
            "--output-dir", str(flows_dir),
        ]
        if args.sample_rate is not None:
            cmd.extend(["--sample-rate", str(args.sample_rate)])
        if args.max_flows is not None:
            cmd.extend(["--max-flows", str(args.max_flows)])
        
        if not run_command(cmd, "Flow Extraction"):
            LOGGER.error("Pipeline failed at flow extraction step")
            sys.exit(1)
    else:
        LOGGER.info("Skipping flow extraction")
    
    # Step 2: Preprocess
    if not args.skip_preprocess:
        cmd = [
            sys.executable,
            "src/preprocess_flows.py",
            "--attack-flows", str(flows_dir / "attack_flows.csv"),
            "--benign-flows", str(flows_dir / "benign_flows.csv"),
            "--output-dir", str(preprocessed_dir),
        ]
        if args.sample_rate is not None:
            cmd.extend(["--sample-rate", str(args.sample_rate)])
        
        if not run_command(cmd, "Data Preprocessing"):
            LOGGER.error("Pipeline failed at preprocessing step")
            sys.exit(1)
    else:
        LOGGER.info("Skipping preprocessing")
    
    # Step 3: Train models
    if not args.skip_train:
        cmd = [
            sys.executable,
            "src/train_unsupervised.py",
            "--X-scaled", str(preprocessed_dir / "X_scaled.csv"),
            "--output-dir", str(models_dir),
            "--contamination", str(args.contamination),
            "--k-min", str(args.k_min),
            "--k-max", str(args.k_max),
            "--ae-epochs", str(args.ae_epochs),
        ]
        
        if not run_command(cmd, "Model Training"):
            LOGGER.error("Pipeline failed at training step")
            sys.exit(1)
    else:
        LOGGER.info("Skipping training")
    
    # Step 4: Evaluate
    if not args.skip_evaluate:
        cmd = [
            sys.executable,
            "src/evaluate_models.py",
            "--predictions", str(models_dir / "predictions.csv"),
            "--X-scaled", str(preprocessed_dir / "X_scaled.csv"),
            "--output-dir", str(plots_dir),
        ]
        
        if not run_command(cmd, "Model Evaluation"):
            LOGGER.error("Pipeline failed at evaluation step")
            sys.exit(1)
    else:
        LOGGER.info("Skipping evaluation")
    
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline Complete!")
    LOGGER.info("=" * 60)
    LOGGER.info(f"All outputs saved to: {output_dir}")
    LOGGER.info(f"  - Flows: {flows_dir}")
    LOGGER.info(f"  - Preprocessed: {preprocessed_dir}")
    LOGGER.info(f"  - Models: {models_dir}")
    LOGGER.info(f"  - Results: {results_dir}")
    LOGGER.info(f"  - Plots: {plots_dir}")


if __name__ == "__main__":
    main()

