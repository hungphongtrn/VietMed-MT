#!/bin/bash

# Directory containing the YAML files
CONFIG_DIR="./zero_shot_cfg"

# Iterate over each YAML file in the configs directory
for config_file in "$CONFIG_DIR"/*.yaml; do
  echo "Processing $config_file..."
  
  # Run the Python script with the current config file
  python zeroshot_boiler.py --config_path="$config_file"
done

echo "All configurations processed."
