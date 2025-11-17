#!/usr/bin/env python3
"""
Flow anomaly detection pipeline with Suricata alert correlation and reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

LOGGER = logging.getLogger("anomaly_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a flow anomaly model with Suricata alert correlation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eve", type=Path, default=Path("data/eve.json"), help="Path to Suricata eve.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full"), help="Directory for artifacts")
    parser.add_argument(
        "--max-flows",
        type=int,
        default=None,
        help="Optional limit on number of flow events to load (useful for demos)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Expected fraction of anomalies for IsolationForest",
    )
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of IsolationForest trees")
    parser.add_argument("--save-model", type=Path, default=None, help="Optional path to persist trained model")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress while parsing eve.json")
    return parser.parse_args()


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    artifacts = {
        "base": base_dir,
        "plots": base_dir / "plots",
        "models": base_dir / "models",
    }
    for path in artifacts.values():
        path.mkdir(parents=True, exist_ok=True)
    return artifacts


def load_suricata_data(
    eve_path: Path, max_flows: Optional[int] = None, show_progress: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    if not eve_path.exists():
        raise FileNotFoundError(f"EVE file not found: {eve_path}")

    flows: List[Dict] = []
    alerts: List[Dict] = []
    flow_ids: set[int] = set()

    LOGGER.info("Reading Suricata events from %s", eve_path)
    with eve_path.open("r") as handle:
        iterator = tqdm(handle, desc="Parsing eve.json") if show_progress else handle
        for line in iterator:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed line")
                continue

            event_type = event.get("event_type")
            if event_type == "flow":
                if max_flows is not None and len(flows) >= max_flows:
                    continue
                flow_data = event.get("flow", {})
                record = {
                    "flow_id": event.get("flow_id"),
                    "src_ip": event.get("src_ip"),
                    "dst_ip": event.get("dest_ip") or event.get("dst_ip"),
                    "src_port": event.get("src_port"),
                    "dst_port": event.get("dest_port"),
                    "proto": event.get("proto"),
                    "app_proto": event.get("app_proto"),
                    "bytes_toserver": flow_data.get("bytes_toserver", 0),
                    "bytes_toclient": flow_data.get("bytes_toclient", 0),
                    "pkts_toserver": flow_data.get("pkts_toserver", 0),
                    "pkts_toclient": flow_data.get("pkts_toclient", 0),
                    "start": flow_data.get("start"),
                    "end": flow_data.get("end"),
                    "state": flow_data.get("state"),
                    "vlan": flow_data.get("vlan"),
                    "flow_age": flow_data.get("age"),
                }
                flows.append(record)
                if record["flow_id"] is not None:
                    flow_ids.add(record["flow_id"])
            elif event_type == "alert":
                flow_id = event.get("flow_id")
                if max_flows is None or not flow_ids or flow_id in flow_ids:
                    alert_data = event.get("alert", {})
                    alerts.append(
                        {
                            "flow_id": flow_id,
                            "timestamp": event.get("timestamp"),
                            "signature": alert_data.get("signature"),
                            "signature_id": alert_data.get("signature_id"),
                            "category": alert_data.get("category"),
                            "severity": alert_data.get("severity"),
                            "src_ip": event.get("src_ip"),
                            "dst_ip": event.get("dest_ip") or event.get("dst_ip"),
                        }
                    )

    flows_df = pd.DataFrame(flows)
    alerts_df = pd.DataFrame(alerts)
    stats = {
        "flows_loaded": len(flows_df),
        "alerts_loaded": len(alerts_df),
        "unique_alert_flows": alerts_df["flow_id"].nunique() if not alerts_df.empty else 0,
    }
    LOGGER.info("Loaded %d flows and %d alerts", stats["flows_loaded"], stats["alerts_loaded"])
    if max_flows and stats["flows_loaded"] < max_flows:
        LOGGER.warning("Requested %d flows but only %d were available", max_flows, stats["flows_loaded"])
    return flows_df, alerts_df, stats


def engineer_features(flows_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if flows_df.empty:
        raise ValueError("No flow events available for feature engineering")

    df = flows_df.copy()
    for col in ("bytes_toserver", "bytes_toclient", "pkts_toserver", "pkts_toclient"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    features = pd.DataFrame(index=df.index)
    features["bytes_total"] = df["bytes_toserver"] + df["bytes_toclient"]
    features["pkts_total"] = df["pkts_toserver"] + df["pkts_toclient"]
    features["bytes_ratio"] = (df["bytes_toserver"] + 1) / (df["bytes_toclient"] + 1)
    features["pkts_ratio"] = (df["pkts_toserver"] + 1) / (df["pkts_toclient"] + 1)
    features["bytes_per_pkt"] = features["bytes_total"] / (features["pkts_total"] + 1)

    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    duration = (df["end"] - df["start"]).dt.total_seconds().fillna(0)
    features["duration"] = duration.clip(lower=0)
    features["bytes_per_sec"] = features["bytes_total"] / (features["duration"] + 1)

    features["proto_encoded"] = df["proto"].fillna("UNKNOWN").astype("category").cat.codes
    features["is_server_port"] = df["dst_port"].isin([22, 23, 25, 53, 80, 110, 143, 443]).astype(int)
    features["is_high_dst_port"] = (
        pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int) >= 1024
    ).astype(int)

    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return features, df


def train_anomaly_model(
    features: pd.DataFrame, contamination: float, n_estimators: int
) -> Tuple[IsolationForest, StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    LOGGER.info("Training IsolationForest on %d samples", len(features))
    model.fit(X_scaled)
    predictions = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)
    return model, scaler, predictions, scores


def correlate_alerts(flows_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    if alerts_df.empty:
        flows_df["alert_count"] = 0
        flows_df["max_alert_severity"] = np.nan
        flows_df["alert_signatures"] = ""
        flows_df["has_alert"] = False
        return flows_df

    summary = (
        alerts_df.groupby("flow_id")
        .agg(
            alert_count=("flow_id", "size"),
            max_alert_severity=("severity", "max"),
            alert_signatures=("signature", lambda x: sorted({s for s in x if s})),
        )
        .reset_index()
    )
    summary["alert_signatures"] = summary["alert_signatures"].apply(lambda sigs: " | ".join(sigs))

    merged = flows_df.merge(summary, how="left", on="flow_id")
    merged["alert_count"] = merged["alert_count"].fillna(0).astype(int)
    merged["has_alert"] = merged["alert_count"] > 0
    merged["max_alert_severity"] = merged["max_alert_severity"]
    merged["alert_signatures"] = merged["alert_signatures"].fillna("")
    return merged


def plot_dataset_characteristics(flows_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8")

    plt.figure(figsize=(8, 4))
    sns.histplot(flows_df["bytes_total"], bins=60, log_scale=(False, True))
    plt.xlabel("Total bytes per flow")
    plt.ylabel("Count (log)")
    plt.title("Flow volume distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "bytes_distribution.png", dpi=200)
    plt.close()

    sample = flows_df.sample(min(5000, len(flows_df)), random_state=42)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=sample,
        x="bytes_toserver",
        y="bytes_toclient",
        hue=sample["is_anomaly"].map({1: "Anomaly", 0: "Normal"}) if "is_anomaly" in sample else None,
        alpha=0.6,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bytes to server (log)")
    plt.ylabel("Bytes to client (log)")
    plt.title("Flow directionality vs anomaly label")
    plt.legend(title="Label", loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / "bytes_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(flows_df["anomaly_score"], bins=60)
    plt.xlabel("IsolationForest score (lower = more anomalous)")
    plt.title("Anomaly score distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "anomaly_scores.png", dpi=200)
    plt.close()

    if "has_alert" in flows_df.columns and "is_anomaly" in flows_df.columns:
        crosstab = flows_df.groupby(["is_anomaly", "has_alert"]).size().reset_index(name="count")
        crosstab["is_anomaly"] = crosstab["is_anomaly"].map({1: "Anomaly", 0: "Normal"})
        crosstab["has_alert"] = crosstab["has_alert"].map({True: "Alert", False: "No alert"})
        plt.figure(figsize=(6, 4))
        sns.barplot(data=crosstab, x="is_anomaly", y="count", hue="has_alert")
        plt.title("Alert overlap with anomaly predictions")
        plt.xlabel("Model prediction")
        plt.ylabel("Flow count")
        plt.tight_layout()
        plt.savefig(plots_dir / "alert_overlap.png", dpi=200)
        plt.close()


def export_summary(flows_df: pd.DataFrame, stats: Dict[str, int], output_dir: Path) -> Dict[str, float]:
    summary = {
        "flows_loaded": stats.get("flows_loaded", len(flows_df)),
        "alerts_loaded": stats.get("alerts_loaded", 0),
        "unique_alert_flows": stats.get("unique_alert_flows", 0),
        "anomalous_flows": int((flows_df["is_anomaly"] == 1).sum()) if "is_anomaly" in flows_df else 0,
        "alerted_flows": int(flows_df.get("has_alert", pd.Series(False)).sum()),
    }
    overlap = flows_df[(flows_df.get("is_anomaly", 0) == 1) & (flows_df.get("has_alert", False))]
    summary["anomaly_alert_overlap"] = int(len(overlap))
    summary["precision_proxy"] = (
        summary["anomaly_alert_overlap"] / summary["anomalous_flows"] if summary["anomalous_flows"] else 0.0
    )
    summary["alert_coverage_proxy"] = (
        summary["anomaly_alert_overlap"] / summary["alerted_flows"] if summary["alerted_flows"] else 0.0
    )

    flows_df.to_csv(output_dir / "flows_with_anomalies.csv", index=False)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Saved summary to %s", output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    artifacts = ensure_output_dirs(args.output_dir)

    flows_df, alerts_df, stats = load_suricata_data(
        args.eve,
        max_flows=args.max_flows,
        show_progress=args.progress,
    )
    if flows_df.empty:
        raise SystemExit("No flows were loaded. Check your eve.json or --max-flows filter.")

    features, flows_df = engineer_features(flows_df)
    model, scaler, preds, scores = train_anomaly_model(
        features,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )

    flows_df["is_anomaly"] = (preds == -1).astype(int)
    flows_df["anomaly_score"] = scores
    flows_df["bytes_total"] = features["bytes_total"]
    flows_df["pkts_total"] = features["pkts_total"]

    flows_df = correlate_alerts(flows_df, alerts_df)

    summary = export_summary(flows_df, stats, artifacts["base"])
    LOGGER.info("Summary: %s", summary)

    if not args.no_plots:
        plot_dataset_characteristics(flows_df, artifacts["base"])

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "scaler": scaler, "feature_names": features.columns.tolist()},
            args.save_model,
        )
        LOGGER.info("Saved model to %s", args.save_model)


if __name__ == "__main__":
    main()


