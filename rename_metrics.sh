#!/bin/bash

# Script to go into each folder in specified directory, then look for metrics folder
# and rename files within it
# Usage: ./rename_metrics.sh <root_directory> <old_filename> <new_filename>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <root_directory> <old_filename> <new_filename>"
    echo "Example: $0 ./mlruns/0/ metrics.json performance_metrics.json"
    exit 1
fi

ROOT_DIR="$1"
OLD_NAME="$2"
NEW_NAME="$3"

echo "Processing each folder in: $ROOT_DIR"
echo "Looking for 'metrics' folders containing: $OLD_NAME"
echo "Renaming to: $NEW_NAME"
echo "----------------------------------------"

# Counter for successful renames
count=0

# Check if root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' does not exist!" >&2
    exit 1
fi

# Process each subdirectory in the root directory
for folder in "$ROOT_DIR"/*/; do
    # Remove trailing slash to get clean folder name
    folder="${folder%/}"
    
    # Check if this is a directory (not a file)
    if [ -d "$folder" ]; then
        metrics_dir="$folder/metrics"
        
        # Check if metrics directory exists
        if [ -d "$metrics_dir" ]; then
            old_file="$metrics_dir/$OLD_NAME"
            new_file="$metrics_dir/$NEW_NAME"
            
            if [ -f "$old_file" ]; then
                echo "Found: $old_file"
                if mv -v "$old_file" "$new_file"; then
                    echo "✓ Renamed to: $new_file"
                    ((count++))
                else
                    echo "✗ Error renaming: $old_file" >&2
                fi
                echo "---"
            fi
        fi
    fi
done

echo "Done! Renamed $count files."