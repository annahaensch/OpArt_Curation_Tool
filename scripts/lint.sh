#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

echo "Checking code style with black..."
python -m black --line-length 100 "${SRC_DIR}"/src/
echo "Success!"

echo "Type checking with mypy..."
mypy --ignore-missing-imports "${SRC_DIR}"/src/
echo "Success!"

# TODO: add pylint back into linter.
echo "Checking code style with pylint..."
python -m pylint "${SRC_DIR}"/src/
echo "Success!"