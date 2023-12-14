#!/bin/bash
#
# Copyright (C) 2023 Cyprien Quéméneur
# For the full license, please refer to the LICENSE file in the root directory of this project.
# For the full copyright notices, please refer to the NOTICE file in the root directory of this project.

# Base link to the official YOLOv7 weights and destination directory
base_url="https://github.com/WongKinYiu/yolov7/releases/download/v0.1"
destination_dir="weights/pretrained"

# Check if the model architecture is provided and check for validity of the argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <yolov7|yolov7x>"
    exit 1
fi
model_type=$1
if [ "$model_type" != "yolov7" ] && [ "$model_type" != "yolov7x" ]; then
    echo "Invalid argument. Please choose either 'yolov7' or 'yolov7x'."
    exit 1
fi

# Define the url and destination path based on the architecture requested, create the destination directory if needed
weights_url="${base_url}/${model_type}_training.pt"
destination_path="${destination_dir}/${model_type}_training.pt"
mkdir -p "$destination_dir"

# Use wget to download the file if it does not exist
if [ ! -f "$destination_path" ]; then
    wget -O "$destination_path" "$weights_url"
    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful! Weights saved to: $destination_path"
    else
        echo "Download failed. Please check the URL and try again."
    fi
else
    echo "File already exists. Skipping download."
fi