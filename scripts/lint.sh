#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

echo "Checking code style with black..."
python -m black --line-length 100 $(git ls-files '*.py')
echo "Success!"

echo "Type checking with mypy..."
mypy --ignore-missing-imports $(git ls-files '*.py')
echo "Success!"

echo "Checking code style with pylint..."
python -m pylint $(git ls-files '*.py')
echo "Success!"