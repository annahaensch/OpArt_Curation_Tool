#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

python -m pip install --upgrade pip
pip install black
pip install black[jupyter]
pip install mypy==0.812
pip install pytest
pip install pytest-cov
          
echo "Checking code style with black..."
python -m black --line-length 100 $(git ls-files '*.py')
echo "Success!"

echo "Type checking with mypy..."
mypy --ignore-missing-imports $(git ls-files '*.py')
echo "Success!"
