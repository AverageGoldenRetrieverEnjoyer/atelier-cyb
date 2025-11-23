# Simple project Makefile

PYTHON ?= python3
SRC := src
DATA_DIR := data
OUTPUT_DIR := outputs

EVE_FULL := $(DATA_DIR)/eve.json
EVE_SAMPLE := $(DATA_DIR)/eve_sample.json

ANOMALY_PIPELINE := $(SRC)/anomaly_pipeline.py

.PHONY: all demo full install merge-data check-integrity fix-corrupted clean

all: demo

install:
	$(PYTHON) -m pip install -r requirements.txt

merge-data:
	@echo ">>> Merging split compressed data files"
	@echo "    This will merge .lz4 PCAP files and .gz EVE JSON files"
	@echo "    from data/gdrive/tpot-backup/logs/ into single files"
	./scripts/merge_data.sh --output $(DATA_DIR)

check-integrity:
	@echo ">>> Checking integrity of lz4 files"
	./scripts/check_lz4_integrity.sh

fix-corrupted:
	@echo ">>> Re-downloading corrupted lz4 files"
	@echo "    Note: Large files may take 10-30 minutes each depending on network speed"
	@echo "    You can run this in the background with: nohup make fix-corrupted &"
	./scripts/redownload_corrupted.sh

demo:
	@echo ">>> Running demo with sample Suricata flows"
	$(PYTHON) $(ANOMALY_PIPELINE) \
		--eve $(EVE_SAMPLE) \
		--output-dir $(OUTPUT_DIR)/demo \
		--max-flows 100000 \
		--progress \
		--save-model $(OUTPUT_DIR)/demo/models/flow_iforest.joblib

full:
	@echo ">>> Running anomaly pipeline on full dataset"
	$(PYTHON) $(ANOMALY_PIPELINE) \
		--eve $(EVE_FULL) \
		--output-dir $(OUTPUT_DIR)/full \
		--progress \
		--save-model $(OUTPUT_DIR)/full/models/flow_iforest.joblib

clean:
	@echo ">>> Removing generated artifacts"
	rm -rf $(OUTPUT_DIR)

import-data:
	@echo ">>> Importing data from Google Drive"
	./scripts/gdrive_import.sh
