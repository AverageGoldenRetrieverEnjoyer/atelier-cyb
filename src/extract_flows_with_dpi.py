#!/usr/bin/env python3
"""
Extract flow-level features from PCAP files with DPI and payload analysis.
Enhanced version with SPLT (Sequence of Packet Length and Time) features
for multiclass attack classification.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

import nfstream
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

# SPLT sequence length for padding
SPLT_SEQUENCE_LENGTH = 10


def categorize_port(port: int) -> str:
    """Categorize port into domain-invariant categories."""
    if port < 1024:
        return "well_known"
    elif port < 49152:
        return "registered"
    else:
        return "ephemeral"


def is_private_ip(ip: str) -> bool:
    """Check if IP is private/internal."""
    if not ip:
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        first = int(parts[0])
        second = int(parts[1])
        # 10.x.x.x, 172.16-31.x.x, 192.168.x.x
        if first == 10:
            return True
        if first == 172 and 16 <= second <= 31:
            return True
        if first == 192 and second == 168:
            return True
        return False
    except ValueError:
        return False


def ip_entropy(ip: str) -> float:
    """Calculate entropy of IP address string (domain-invariant feature)."""
    if not ip:
        return 0.0
    # Hash-based entropy approximation
    h = hashlib.md5(ip.encode()).hexdigest()
    return sum(int(c, 16) for c in h[:8]) / 120.0  # Normalize to ~0-1


def extract_flows_with_dpi(
    pcap_path: Path,
    output_path: Path,
    sample_rate: Optional[float] = None,
    max_flows: Optional[int] = None,
    splt_length: int = SPLT_SEQUENCE_LENGTH,
) -> int:
    """
    Extract flow features from PCAP file with DPI and SPLT analysis.
    
    Args:
        pcap_path: Path to input PCAP file
        output_path: Path to output CSV file
        sample_rate: Optional sampling rate (0.0-1.0)
        max_flows: Optional maximum number of flows
        splt_length: Number of packets to analyze in SPLT
    
    Returns:
        Number of flows extracted
    """
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    LOGGER.info(f"Processing PCAP: {pcap_path}")
    LOGGER.info(f"File size: {pcap_path.stat().st_size / (1024**3):.2f} GB")

    # Configure nfstream with DPI and SPLT analysis
    nfs = nfstream.NFStreamer(
        source=str(pcap_path),
        statistical_analysis=True,      # Enable statistical features
        splt_analysis=splt_length,      # Analyze first N packets
        n_dissections=20,               # Enable L7 protocol detection
        accounting_mode=0,              # Standard accounting
    )

    flows = []
    flow_count = 0

    LOGGER.info(f"Extracting flows with DPI (splt_analysis={splt_length})...")
    
    for flow in tqdm(nfs, desc="Processing flows"):
        # Apply sampling if specified
        if sample_rate is not None and flow_count > 0:
            import random
            if random.random() > sample_rate:
                continue

        # === DOMAIN-INVARIANT FEATURES ===
        src_private = is_private_ip(flow.src_ip)
        dst_private = is_private_ip(flow.dst_ip)
        src_port_cat = categorize_port(flow.src_port)
        dst_port_cat = categorize_port(flow.dst_port)
        
        # === BASIC FLOW FEATURES ===
        total_packets = flow.bidirectional_packets
        total_bytes = flow.bidirectional_bytes
        duration_s = flow.bidirectional_duration_ms / 1000.0 + 1e-9
        
        flow_dict = {
            # Identifiers (for correlation, not training)
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            
            # Domain-invariant features
            "src_is_private": int(src_private),
            "dst_is_private": int(dst_private),
            "both_private": int(src_private and dst_private),
            "src_port_category": src_port_cat,
            "dst_port_category": dst_port_cat,
            "src_ip_entropy": ip_entropy(flow.src_ip),
            "dst_ip_entropy": ip_entropy(flow.dst_ip),
            
            # Volume features (log-transformed for robustness)
            "log_packets": np.log1p(total_packets),
            "log_bytes": np.log1p(total_bytes),
            "log_packets_src2dst": np.log1p(flow.src2dst_packets),
            "log_bytes_src2dst": np.log1p(flow.src2dst_bytes),
            "log_packets_dst2src": np.log1p(flow.dst2src_packets),
            "log_bytes_dst2src": np.log1p(flow.dst2src_bytes),
            
            # Ratio features (network-invariant)
            "packets_ratio": flow.src2dst_packets / (total_packets + 1),
            "bytes_ratio": flow.src2dst_bytes / (total_bytes + 1),
            
            # Duration features
            "log_duration": np.log1p(duration_s),
            
            # Rate features
            "packet_rate": total_packets / duration_s,
            "byte_rate": total_bytes / duration_s,
            
            # Inter-arrival time statistics
            "mean_iat_ms": getattr(flow, "bidirectional_mean_iat_ms", 0),
            "std_iat_ms": getattr(flow, "bidirectional_stddev_iat_ms", 0),
            "min_iat_ms": getattr(flow, "bidirectional_min_iat_ms", 0),
            "max_iat_ms": getattr(flow, "bidirectional_max_iat_ms", 0),
            
            # Packet size statistics
            "mean_ps": getattr(flow, "bidirectional_mean_ps", 0),
            "std_ps": getattr(flow, "bidirectional_stddev_ps", 0),
            "min_ps": getattr(flow, "bidirectional_min_ps", 0),
            "max_ps": getattr(flow, "bidirectional_max_ps", 0),
            
            # TCP flags
            "syn_count": getattr(flow, "src2dst_syn_count", 0),
            "ack_count": getattr(flow, "src2dst_ack_count", 0),
            "fin_count": getattr(flow, "src2dst_fin_count", 0),
            "rst_count": getattr(flow, "src2dst_rst_count", 0),
            "psh_count": getattr(flow, "src2dst_psh_count", 0),
            "urg_count": getattr(flow, "src2dst_urg_count", 0),
            
            # L7 Application info (category for domain invariance)
            "application_name": getattr(flow, "application_name", "unknown"),
            "application_category": getattr(flow, "application_category_name", "unknown"),
            
            # TLS metadata (if available)
            "requested_server_name": getattr(flow, "requested_server_name", ""),
            "client_fingerprint": getattr(flow, "client_fingerprint", ""),
            
            # Timestamps
            "first_seen_ms": flow.bidirectional_first_seen_ms,
            "last_seen_ms": flow.bidirectional_last_seen_ms,
        }
        
        # === SPLT FEATURES (Sequence of Packet Length and Time) ===
        # These are crucial for multiclass attack classification
        splt_direction = getattr(flow, "splt_direction", [])
        splt_ps = getattr(flow, "splt_ps", [])
        splt_piat_ms = getattr(flow, "splt_piat_ms", [])
        
        # Pad/truncate to fixed length
        for i in range(splt_length):
            flow_dict[f"splt_dir_{i}"] = splt_direction[i] if i < len(splt_direction) else 0
            flow_dict[f"splt_ps_{i}"] = splt_ps[i] if i < len(splt_ps) else 0
            flow_dict[f"splt_piat_{i}"] = splt_piat_ms[i] if i < len(splt_piat_ms) else 0
        
        # SPLT aggregate features
        if splt_ps:
            flow_dict["splt_ps_mean"] = np.mean(splt_ps)
            flow_dict["splt_ps_std"] = np.std(splt_ps)
            flow_dict["splt_ps_min"] = np.min(splt_ps)
            flow_dict["splt_ps_max"] = np.max(splt_ps)
        else:
            flow_dict["splt_ps_mean"] = 0
            flow_dict["splt_ps_std"] = 0
            flow_dict["splt_ps_min"] = 0
            flow_dict["splt_ps_max"] = 0
            
        if splt_piat_ms:
            flow_dict["splt_piat_mean"] = np.mean(splt_piat_ms)
            flow_dict["splt_piat_std"] = np.std(splt_piat_ms)
        else:
            flow_dict["splt_piat_mean"] = 0
            flow_dict["splt_piat_std"] = 0
        
        # Direction entropy (attack pattern indicator)
        if splt_direction:
            dir_counts = np.bincount(splt_direction, minlength=2)
            dir_probs = dir_counts / (len(splt_direction) + 1e-9)
            dir_entropy = -np.sum(dir_probs * np.log2(dir_probs + 1e-9))
            flow_dict["splt_dir_entropy"] = dir_entropy
        else:
            flow_dict["splt_dir_entropy"] = 0

        flows.append(flow_dict)
        flow_count += 1

        # Check max_flows limit
        if max_flows is not None and flow_count >= max_flows:
            LOGGER.info(f"Reached max_flows limit: {max_flows}")
            break

        # Periodic save for memory efficiency
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
        description="Extract flow features with DPI and SPLT analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pcap", type=Path, required=True, help="Path to PCAP file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--label", type=str, default=None, 
                        help="Label to add to all flows (e.g., 'benign', 'attack')")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Sampling rate (0.0-1.0)")
    parser.add_argument("--max-flows", type=int, default=None,
                        help="Maximum flows to extract")
    parser.add_argument("--splt-length", type=int, default=SPLT_SEQUENCE_LENGTH,
                        help="SPLT sequence length")

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Extract flows
    flow_count = extract_flows_with_dpi(
        args.pcap,
        args.output,
        args.sample_rate,
        args.max_flows,
        args.splt_length,
    )
    
    # Add label if specified
    if args.label and flow_count > 0:
        LOGGER.info(f"Adding label '{args.label}' to all flows...")
        df = pd.read_csv(args.output)
        df["label"] = args.label
        df.to_csv(args.output, index=False)
        LOGGER.info("Labels added.")
    
    LOGGER.info("=" * 60)
    LOGGER.info(f"Extraction complete: {flow_count:,} flows")
    LOGGER.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
