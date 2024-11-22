#!/bin/bash

# Exit script if any command fails
set -e

# Clean up build artifacts
echo "Cleaning up build artifacts..."
rm -rf dist/ *.egg-info/
echo "Build artifacts cleaned up."

echo "Process completed successfully."
