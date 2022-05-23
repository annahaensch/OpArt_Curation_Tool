PYTHONPATH := $(CURDIR)
SHELL := /bin/bash

# Install required python libraries
install_dependencies: 
	conda install pip
	pip install -U -r requirements.txt

preprocess_data:
	python $(PYTHONPATH)/src/preprocessing.py