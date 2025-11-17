# Simple project Makefile

PYTHON ?= python3
SRC := src
DATA_DIR := data
OUTPUT_DIR := outputs

EVE_FULL := $(DATA_DIR)/eve.json
EVE_SAMPLE := $(DATA_DIR)/eve_sample.json

ANOMALY_PIPELINE := $(SRC)/anomaly_pipeline.py

.PHONY: all demo full install clean

all: demo

install:
	$(PYTHON) -m pip install -r requirements.txt

demo:
	@echo ">>> Running demo with sample Suricata flows"
	$(PYTHON) $(ANOMALY_PIPELINE) \
		--eve $(EVE_SAMPLE) \
		--output-dir $(OUTPUT_DIR)/demo \
		--max-flows 100000 \
		--progress
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
