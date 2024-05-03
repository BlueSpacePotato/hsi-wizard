#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run pydocstyle
if pydocstyle hsi_wizard; then
# Run flake8
  flake8 hsi_wizard
else
    echo "pydocstyle check failed, skipping flake8."
fi

# Deactivate the virtual environment
deactivate