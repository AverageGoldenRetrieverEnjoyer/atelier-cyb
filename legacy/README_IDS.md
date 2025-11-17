# Memory-Efficient IDS System

A two-stage Intrusion Detection System (IDS) designed for low-memory environments, using Suricata flow data and PCAP files.

## Architecture

### Model 1: Flow-based Anomaly Detection
- **Purpose**: Detect anomalous network flows using statistical features
- **Algorithm**: Isolation Forest (unsupervised)
- **Features**: 
  - Volume metrics (bytes, packets, ratios)
  - Duration and rate features
  - Protocol and port patterns
  - Flow characteristics

### Model 2: Payload-based Attack Classification
- **Purpose**: Classify attack types from PCAP payloads
- **Memory Optimization**: 
  - Fuzzy hashing (ssdeep) instead of raw payloads
  - SQLite database for hash storage
  - Batch processing of PCAP files
  - Streaming packet processing
- **Attack Types Detected**:
  - SQL Injection
  - XSS (Cross-Site Scripting)
  - Command Injection
  - Path Traversal
  - Brute Force
  - Exploits
  - Scanning

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Training mode (first run)
python run_ids.py eve.json pcaps_tpot --train

# Inference mode (subsequent runs)
python run_ids.py eve.json pcaps_tpot
```

### Advanced Usage

```bash
# Using the main module directly
python ids_system.py --eve eve.json --pcap-dir pcaps_tpot --train --output results.csv
```

### Python API

```python
from ids_system import IDSSystem

# Initialize
ids = IDSSystem()

# Run detection
results = ids.run_detection(
    eve_json_path="eve.json",
    pcap_dir="pcaps_tpot",
    train_mode=True  # False for inference
)

# Get summary
summary = ids.get_summary(results)
print(summary)
```

## Output

The system generates:
1. **ids_results.csv**: Complete flow analysis with:
   - Flow metadata (IPs, ports, protocol)
   - Anomaly detection results (`is_anomaly`, `anomaly_score`)
   - Attack classification (`attack_type`, `attack_confidence`)
   - Combined threat score (`threat_score`)

2. **fuzzy_hashes.db**: SQLite database storing:
   - Fuzzy hashes of all processed payloads
   - Attack classifications
   - Flow associations

3. **flow_anomaly_model.joblib**: Trained anomaly detection model

## Memory Optimization Techniques

1. **Fuzzy Hashing**: Uses ssdeep to create compact signatures instead of storing raw payloads
2. **Batch Processing**: Processes PCAP files in small batches
3. **Streaming**: Reads packets one at a time, not loading entire PCAPs
4. **SQLite Storage**: Efficient database for hash lookups
5. **Limited Packet Processing**: Processes max 1000 packets per PCAP by default

## Configuration

### Adjusting Memory Usage

Edit `ids_system.py`:

```python
# In PayloadAttackClassifier.extract_payload_from_pcap()
max_packets: int = 1000  # Reduce for lower memory

# In PayloadAttackClassifier.process_pcap_batch()
batch_size: int = 5  # Reduce for lower memory
```

### Adjusting Detection Sensitivity

```python
# In FlowAnomalyDetector.__init__()
contamination=0.01  # Lower = fewer anomalies detected
```

## Performance Tips

1. **First Run**: Use `--train` to build the model
2. **Subsequent Runs**: Skip training for faster inference
3. **Large Datasets**: Process PCAPs in separate batches
4. **Memory Monitoring**: Watch memory usage and adjust batch sizes

## File Structure

```
pourtpot/
├── ids_system.py          # Main IDS system
├── run_ids.py             # Simple wrapper script
├── fuzzy_hashes.db        # Generated: Fuzzy hash database
├── flow_anomaly_model.joblib  # Generated: Trained model
├── ids_results.csv        # Generated: Detection results
└── ids_system.log         # Generated: System logs
```

## Troubleshooting

### Out of Memory
- Reduce `max_packets` in `extract_payload_from_pcap()`
- Reduce `batch_size` in `process_pcap_batch()`
- Process PCAPs separately

### Slow Processing
- Skip PCAP processing if only flow analysis needed
- Use inference mode (skip `--train`)
- Reduce number of PCAP files processed

### Missing Dependencies
```bash
pip install ssdeep  # May require system libraries
# On Ubuntu/Debian: sudo apt-get install libfuzzy-dev
```

## Future Enhancements

- Deep learning models for payload analysis
- Real-time streaming detection
- Integration with SIEM systems
- Custom attack pattern definitions
- Performance metrics and visualization

