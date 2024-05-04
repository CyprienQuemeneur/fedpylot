#!/bin/bash
# FedPylot by Cyprien Quéméneur, GPL-3.0 license
# Example usage: bash weights/get_weights.sh yolov7

# Base link to the official YOLOv7 weights and destination directory
base_url="https://github.com/WongKinYiu/yolov7/releases/download/v0.1"
destination_dir="weights/yolov7"

# Check if the model architecture is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <yolov7-tiny|yolov7|yolov7x|yolov7-w6|yolov7-e6|yolov7-d6|yolov7-e6e>"
    exit 1
fi
input=$1

# Check for validity of the argument
case $input in
    "yolov7-tiny"|"yolov7"|"yolov7x"|"yolov7-w6"|"yolov7-e6"|"yolov7-d6"|"yolov7-e6e")
        # Define the url and destination path based on the architecture requested, create the destination directory
        if [ $input = "yolov7-tiny" ]; then
            model="${input}.pt"
        else
            model="${input}_training.pt"
        fi
        weights_url="${base_url}/${model}"
        destination_path="${destination_dir}/${model}"
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
        ;;
    *)
        echo "Invalid argument. Available: yolov7-tiny, yolov7, yolov7x, yolov7-w6, yolov7-e6, yolov7-d6, yolov7-e6e."
        exit 1
esac
