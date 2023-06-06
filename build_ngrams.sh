#!/bin/bash

# Check if a directory is provided as an argument
if [ $# -eq 0 ]; then
  echo "Please provide a directory as an argument."
  exit 1
fi

# Directory to process
directory="$1"

# Iterate over each file in the specified directory
for file in "$directory"/*.srt; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    ./build_ngram.sh "$file" "${filename%.*}"
  fi
done

