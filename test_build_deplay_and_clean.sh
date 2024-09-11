#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run pydocstyle
if ! pydocstyle wizard; then
    echo "pydocstyle check failed. Exiting."
    deactivate
    exit 1
fi

# Run flake8
if ! flake8 wizard; then
    echo "flake8 check failed. Exiting."
    deactivate
    exit 1
fi

# Deactivate the virtual environment
deactivate

# Exit script if any command fails
set -e

# Ask for the Git tag
echo "Enter the new Git tag (e.g., v1.0.0):"
read tag
git tag "$tag"
git push --tags
echo "Git tag $tag has been created and pushed."

# Build the Python package
echo "Building the package..."
python3 -m build
echo "Package build complete."

# Upload the package to PyPI
echo "Uploading package to PyPI..."
twine upload dist/*
echo "Package uploaded to PyPI."

# Trigger a Read the Docs build (optional)
echo "Triggering Read the Docs build..."
# Add your custom webhook or integration here if needed
echo "Read the Docs build triggered (check RTD for confirmation)."

# Clean up build artifacts
echo "Cleaning up build artifacts..."
rm -rf dist/ *.egg-info/
echo "Build artifacts cleaned up."

echo "Process completed successfully."
