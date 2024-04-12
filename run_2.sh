#!/bin/bash

# Directory containing the YAML files
CONFIG_DIR="./configs_2"

# Iterate over each YAML file in the configs directory
for config_file in "$CONFIG_DIR"/*.yaml; do
  echo "Processing $config_file..."
  
  # Run the Python script with the current config file
  python boiler.py --config_path="$config_file"
done

echo "All configurations processed."
