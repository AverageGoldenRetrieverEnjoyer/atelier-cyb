import os
import glob
import json
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scapy.all import rdpcap

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("pipeline.log"),
                        logging.StreamHandler()
                    ])

# -------------------------------
# 1️⃣ LOAD SURICATA FLOWS
# -------------------------------
def load_suricata_flows(eve_json_path):
    logging.info(f"Loading Suricata flows from {eve_json_path}")
    flows = []
    with open(eve_json_path, 'r') as f:
        for line in f:
            try:
                evt = json.loads(line)
                if evt.get("event_type") == "flow":
                    flow_data = evt["flow"]
                    flows.append({
                        "src_ip": evt.get("src_ip"),
                        "dst_ip": evt.get("dest_ip"),
                        "src_port": evt.get("src_port"),
                        "dst_port": evt.get("dest_port"),
                        "proto": evt.get("proto"),
                        "bytes_toserver": flow_data.get("bytes_toserver", 0),
                        "bytes_toclient": flow_data.get("bytes_toclient", 0),
                        "pkts_toserver": flow_data.get("pkts_toserver", 0),
                        "pkts_toclient": flow_data.get("pkts_toclient", 0),
                        "start": flow_data.get("start"),
                        "end": flow_data.get("end"),
                        "ja3": evt.get("ja3", np.nan)
                    })
            except Exception as e:
                logging.warning(f"Skipping line due to parse error: {e}")
    df = pd.DataFrame(flows)
    logging.info(f"Loaded {len(df)} flows")
    return df

# -------------------------------
# 2️⃣ PREPROCESS FEATURES
# -------------------------------
def preprocess_features(df):
    logging.info("Preprocessing features for ML")
    feature_cols = ["bytes_toserver", "bytes_toclient",
                    "pkts_toserver", "pkts_toclient"]
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature scaling complete")
    return X_scaled, scaler

# -------------------------------
# 3️⃣ ANOMALY DETECTION (Stage 1)
# -------------------------------
def train_anomaly_model(X):
    logging.info("Training IsolationForest for anomaly detection")
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(X)
    logging.info("IsolationForest training complete")
    return iso

def predict_anomaly(model, X):
    logging.info("Predicting anomalies on flows")
    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    return preds

# -------------------------------
# 4️⃣ PCAP PAYLOAD DATASET
# -------------------------------
class PCAPPayloadDataset(Dataset):
    def __init__(self, pcap_files, max_bytes=1500):
        self.data = []
        self.labels = []
        self.max_bytes = max_bytes
        for f in pcap_files:
            logging.info(f"Reading PCAP {f}")
            packets = rdpcap(f)
            for pkt in packets:
                raw = bytes(pkt)[:max_bytes]
                if len(raw) < max_bytes:
                    raw += b'\x00' * (max_bytes - len(raw))
                self.data.append(np.frombuffer(raw, dtype=np.uint8))
                self.labels.append(0)  # placeholder
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        logging.info(f"Payload dataset size: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -------------------------------
# 5️⃣ CNN MODEL
# -------------------------------
class PayloadCNN(nn.Module):
    def __init__(self, input_len=1500, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        conv_out_size = ((input_len - 4)//2 - 4)//2 * 64
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_payload_model(dataset, epochs=5, batch_size=32):
    logging.info("Training CNN on payload dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PayloadCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
    logging.info("CNN training complete")
    return model

# -------------------------------
# 6️⃣ MAIN PIPELINE
# -------------------------------
if __name__ == "__main__":
    logging.info("=== START PIPELINE ===")

    # Paths
    eve_json_path = "eve.json"  # Suricata flows
    pcap_dir = "pcaps_tpot"    # TPOT PCAPs

    # Load flows
    flows = load_suricata_flows(eve_json_path)

    # Preprocess features
    X_scaled, scaler = preprocess_features(flows)

    # Stage 1: anomaly detection
    anomaly_model = train_anomaly_model(X_scaled)
    preds = predict_anomaly(anomaly_model, X_scaled)
    flows["anomaly"] = preds
    logging.info(f"Detected {np.sum(preds==-1)} anomalous flows out of {len(preds)}")

    # Stage 2: payload deep learning
    pcap_files = glob.glob(os.path.join(pcap_dir, "*.pcap"))
    payload_dataset = PCAPPayloadDataset(pcap_files)
    payload_model = train_payload_model(payload_dataset, epochs=3)  # keep short for demo

    logging.info("=== PIPELINE COMPLETE ===")
    flows.to_csv("flows_anomaly.csv", index=False)
    logging.info("Flows with anomaly labels saved to flows_anomaly.csv")
