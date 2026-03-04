#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate


# Ensure pydocstyle is installed
if ! command -v pydocstyle >/dev/null 2>&1; then
    echo "pydocstyle not found. Installing..."
    pip install pydocstyle
fi

if ! command -v flake8 >/dev/null 2>&1; then
    echo "pydocstyle not found. Installing..."
    pip install flake8
fi

# Run pydocstyle
if pydocstyle wizard; then
# Run flake8
  flake8 wizard
else
    echo "pydocstyle check failed, skipping flake8."
    break
fi

pytest --cov=wizard --cov-report=term-missing

# Deactivate the virtual environment
deactivate

# Cleanup temporary coverage files
find . -maxdepth 1 -type f -name ".coverage.*" -delete
