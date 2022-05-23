PYTHONPATH := $(CURDIR)
SHELL := /bin/bash

# Install required python libraries
install_dependencies: 
	conda install pip
	pip install -U -r requirements.txt
	python3 -m pip install types-requests

# Preprocess data
preprocess_data:
	python $(PYTHONPATH)/src/preprocessing.py
