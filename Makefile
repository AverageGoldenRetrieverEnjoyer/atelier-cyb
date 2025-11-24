# Simple project Makefile

PYTHON ?= python3
SRC := src
DATA_DIR := data
OUTPUT_DIR := outputs

EVE_FULL := $(DATA_DIR)/eve.json
EVE_SAMPLE := $(DATA_DIR)/eve_sample.json

ANOMALY_PIPELINE := $(SRC)/anomaly_pipeline.py

.PHONY: all demo full install merge-data check-integrity fix-corrupted connect clean \
	unsupervised-pipeline extract-flows preprocess-flows train-unsupervised evaluate-unsupervised

all: install prepare_data

install:
	@echo ">>> Installing dependencies"
	$(PYTHON) -m venv venv	
	source venv/bin/activate	
	@echo ">>> Installing dependencies"
	$(PYTHON) -m pip install -r requirements.txt

merge-data:
	@echo ">>> Merging split compressed data files"
	@echo "    This will merge .lz4 PCAP files and .gz EVE JSON files"
	@echo "    from data/logs/pcap/ and data/logs/eve/ into single files"
	./scripts/merge_data.sh --output $(DATA_DIR)

check-integrity:
	@echo ">>> Checking integrity of lz4 files"
	./scripts/check_lz4_integrity.sh

fix-corrupted:
	@echo ">>> Re-downloading corrupted lz4 files"
	@echo "    Note: Large files may take 10-30 minutes each depending on network speed"
	@echo "    You can run this in the background with: nohup make fix-corrupted &"
	./scripts/redownload_corrupted.sh

demo_training:
	@echo ">>> Running demo with sample Suricata flows"
	$(PYTHON) $(ANOMALY_PIPELINE) \
		--eve $(EVE_SAMPLE) \
		--output-dir $(OUTPUT_DIR)/demo \
		--max-flows 100000 \
		--progress \
		--save-model $(OUTPUT_DIR)/demo/models/flow_iforest.joblib

full_training:
	@echo ">>> Running anomaly pipeline on full dataset"
	$(PYTHON) $(ANOMALY_PIPELINE) \
		--eve $(EVE_FULL) \
		--output-dir $(OUTPUT_DIR)/full \
		--progress \
		--save-model $(OUTPUT_DIR)/full/models/flow_iforest.joblib

connect:
	@echo ">>> Connecting to VPN and SSH"
	@echo "    This will:"
	@echo "    1. Check/install WireGuard"
	@echo "    2. Connect to VPN using access_keys/*.conf"
	@echo "    3. SSH to host using access_keys/*.pem"
	./scripts/connect_vpn_ssh.sh

clean:
	@echo ">>> Removing generated artifacts"
	rm -rf $(OUTPUT_DIR)

import-data:
	@echo ">>> Importing data from Google Drive"
	./scripts/gdrive_import.sh

download-benin:
	@echo ">>> Downloading Benin dataset"
	./scripts/download_benin.sh

prepare_data: import-data check-integrity fix-corrupted download-benin merge-data

# Unsupervised ML Pipeline
ATTACK_PCAP := $(DATA_DIR)/attack_data.pcap
BENIGN_PCAP := $(DATA_DIR)/benign_data.pcap

unsupervised-pipeline:
	@echo ">>> Running complete unsupervised intrusion detection pipeline"
	@echo "    This will: extract flows → preprocess → train → evaluate"
	$(PYTHON) $(SRC)/run_unsupervised_pipeline.py \
		--attack-pcap $(ATTACK_PCAP) \
		--benign-pcap $(BENIGN_PCAP) \
		--output-base $(OUTPUT_DIR)

extract-flows:
	@echo ">>> Extracting flows from PCAP files"
	$(PYTHON) $(SRC)/pcap_to_flows.py \
		--attack-pcap $(ATTACK_PCAP) \
		--benign-pcap $(BENIGN_PCAP) \
		--output-dir $(OUTPUT_DIR)/flows

preprocess-flows:
	@echo ">>> Preprocessing flow data"
	$(PYTHON) $(SRC)/preprocess_flows.py \
		--attack-flows $(OUTPUT_DIR)/flows/attack_flows.csv \
		--benign-flows $(OUTPUT_DIR)/flows/benign_flows.csv \
		--output-dir $(OUTPUT_DIR)/preprocessed

train-unsupervised:
	@echo ">>> Training unsupervised models (IF, K-Means, Autoencoder)"
	$(PYTHON) $(SRC)/train_unsupervised.py \
		--X-scaled $(OUTPUT_DIR)/preprocessed/X_scaled.csv \
		--output-dir $(OUTPUT_DIR)/models

evaluate-unsupervised:
	@echo ">>> Evaluating and visualizing models"
	$(PYTHON) $(SRC)/evaluate_models.py \
		--predictions $(OUTPUT_DIR)/models/predictions.csv \
		--X-scaled $(OUTPUT_DIR)/preprocessed/X_scaled.csv \
		--preprocessor $(OUTPUT_DIR)/preprocessed/preprocessor.joblib \
		--models-dir $(OUTPUT_DIR)/models \
		--output-dir $(OUTPUT_DIR)/plots