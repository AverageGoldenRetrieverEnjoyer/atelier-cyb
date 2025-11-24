#!/usr/bin/env python3
"""
Extract flow-level features from PCAP files using nfstream.
Memory-efficient processing with chunking support.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import nfstream
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


def extract_flows_from_pcap(
    pcap_path: Path,
    output_path: Path,
    sample_rate: Optional[float] = None,
    max_flows: Optional[int] = None,
) -> int:
    """
    Extract flow features from PCAP file using nfstream.
    
    Args:
        pcap_path: Path to input PCAP file
        output_path: Path to output CSV file
        sample_rate: Optional sampling rate (0.0-1.0) for downsampling
        max_flows: Optional maximum number of flows to extract
    
    Returns:
        Number of flows extracted
    """
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    LOGGER.info(f"Processing PCAP: {pcap_path}")
    LOGGER.info(f"File size: {pcap_path.stat().st_size / (1024**3):.2f} GB")

    # Configure nfstream with comprehensive features
    nfs = nfstream.NFStreamer(
        source=str(pcap_path),
        statistical_analysis=True,  # Enable statistical features
        splt_analysis=0,  # Disable payload analysis (memory efficient)
        n_dissections=0,  # Disable deep packet inspection (faster)
        system_visibility_mode=0,  # Disable system visibility (memory efficient)
    )

    flows = []
    flow_count = 0

    LOGGER.info("Extracting flows...")
    for flow in tqdm(nfs, desc="Processing flows"):
        # Apply sampling if specified
        if sample_rate is not None and flow_count > 0:
            import random
            if random.random() > sample_rate:
                continue

        # Extract comprehensive flow features
        flow_dict = {
            # Basic flow identifiers
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "vlan_id": flow.vlan_id if hasattr(flow, "vlan_id") else 0,

            # Volume features
            "packets": flow.bidirectional_packets,
            "bytes": flow.bidirectional_bytes,
            "packets_src2dst": flow.src2dst_packets,
            "bytes_src2dst": flow.src2dst_bytes,
            "packets_dst2src": flow.dst2src_packets,
            "bytes_dst2src": flow.dst2src_bytes,

            # Duration features
            "duration": flow.bidirectional_duration_ms / 1000.0,  # Convert to seconds
            "duration_src2dst_ms": flow.src2dst_duration_ms,
            "duration_dst2src_ms": flow.dst2src_duration_ms,

            # Rate features
            "packet_rate": flow.bidirectional_packets / (flow.bidirectional_duration_ms / 1000.0 + 1e-6),
            "byte_rate": flow.bidirectional_bytes / (flow.bidirectional_duration_ms / 1000.0 + 1e-6),

            # Inter-arrival time statistics
            "mean_iat": flow.bidirectional_mean_iat_ms if hasattr(flow, "bidirectional_mean_iat_ms") else 0,
            "std_iat": flow.bidirectional_stddev_iat_ms if hasattr(flow, "bidirectional_stddev_iat_ms") else 0,
            "min_iat": flow.bidirectional_min_iat_ms if hasattr(flow, "bidirectional_min_iat_ms") else 0,
            "max_iat": flow.bidirectional_max_iat_ms if hasattr(flow, "bidirectional_max_iat_ms") else 0,

            # Packet size statistics
            "mean_packet_size": flow.bidirectional_mean_ps if hasattr(flow, "bidirectional_mean_ps") else 0,
            "std_packet_size": flow.bidirectional_stddev_ps if hasattr(flow, "bidirectional_stddev_ps") else 0,
            "min_packet_size": flow.bidirectional_min_ps if hasattr(flow, "bidirectional_min_ps") else 0,
            "max_packet_size": flow.bidirectional_max_ps if hasattr(flow, "bidirectional_max_ps") else 0,

            # TCP flags (if TCP)
            "tcp_flags": flow.tcp_flags if hasattr(flow, "tcp_flags") else 0,
            "syn_count": flow.src2dst_syn_count if hasattr(flow, "src2dst_syn_count") else 0,
            "ack_count": flow.src2dst_ack_count if hasattr(flow, "src2dst_ack_count") else 0,
            "fin_count": flow.src2dst_fin_count if hasattr(flow, "src2dst_fin_count") else 0,
            "rst_count": flow.src2dst_rst_count if hasattr(flow, "src2dst_rst_count") else 0,

            # Application protocol
            "application_name": flow.application_name if hasattr(flow, "application_name") else "unknown",
            "application_category_name": flow.application_category_name if hasattr(flow, "application_category_name") else "unknown",

            # Flow direction
            "src2dst_packets_ratio": flow.src2dst_packets / (flow.bidirectional_packets + 1),
            "src2dst_bytes_ratio": flow.src2dst_bytes / (flow.bidirectional_bytes + 1),

            # Timestamps
            "first_seen": flow.bidirectional_first_seen_ms,
            "last_seen": flow.bidirectional_last_seen_ms,
        }

        flows.append(flow_dict)
        flow_count += 1

        # Check max_flows limit
        if max_flows is not None and flow_count >= max_flows:
            LOGGER.info(f"Reached max_flows limit: {max_flows}")
            break

        # Periodic save for memory efficiency (every 10k flows)
        if len(flows) >= 10000:
            df_chunk = pd.DataFrame(flows)
            if output_path.exists():
                df_chunk.to_csv(output_path, mode="a", header=False, index=False)
            else:
                df_chunk.to_csv(output_path, mode="w", header=True, index=False)
            flows = []
            LOGGER.debug(f"Saved chunk, total flows: {flow_count}")

    # Save remaining flows
    if flows:
        df = pd.DataFrame(flows)
        if output_path.exists():
            df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            df.to_csv(output_path, mode="w", header=True, index=False)

    LOGGER.info(f"Extracted {flow_count} flows to {output_path}")
    return flow_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract flow features from PCAP files using nfstream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attack-pcap", type=Path, required=True, help="Path to attack PCAP file")
    parser.add_argument("--benign-pcap", type=Path, required=True, help="Path to benign PCAP file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for flow CSVs")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Sampling rate (0.0-1.0) for downsampling. None = no sampling",
    )
    parser.add_argument(
        "--max-flows",
        type=int,
        default=None,
        help="Maximum number of flows to extract per file. None = all flows",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract attack flows
    attack_output = args.output_dir / "attack_flows.csv"
    LOGGER.info("=" * 60)
    LOGGER.info("Extracting ATTACK flows")
    LOGGER.info("=" * 60)
    attack_count = extract_flows_from_pcap(
        args.attack_pcap, attack_output, args.sample_rate, args.max_flows
    )

    # Extract benign flows
    benign_output = args.output_dir / "benign_flows.csv"
    LOGGER.info("=" * 60)
    LOGGER.info("Extracting BENIGN flows")
    LOGGER.info("=" * 60)
    benign_count = extract_flows_from_pcap(
        args.benign_pcap, benign_output, args.sample_rate, args.max_flows
    )

    LOGGER.info("=" * 60)
    LOGGER.info("Extraction Summary")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Attack flows: {attack_count:,}")
    LOGGER.info(f"Benign flows: {benign_count:,}")
    LOGGER.info(f"Total flows: {attack_count + benign_count:,}")
    LOGGER.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

