#!/bin/bash

# Script to extract all tar.gz files in the dataset directory

# Set the main directory
DATASET_DIR="dataset"

# Find all tar.gz files and extract them
find "$DATASET_DIR" -name "*.tar.gz" -type f | while read -r file; do
    echo "Extracting: $file"
    
    # Get the directory containing the file
    dir=$(dirname "$file")
    
    # Extract the file in its current directory
    tar -xzf "$file" -C "$dir"
    
    echo "Extracted: $file"
done

echo "All tar.gz files have been extracted."