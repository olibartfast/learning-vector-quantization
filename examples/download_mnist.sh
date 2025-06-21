#!/bin/bash

# MNIST Dataset Download Script
# This script downloads the MNIST dataset for testing the LVQ network

set -e  # Exit on any error

echo "Downloading MNIST Dataset..."

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    mkdir data
fi

if [ ! -d "data/mnist" ]; then
    mkdir data/mnist
fi

cd data/mnist

# MNIST dataset URLs
TRAIN_IMAGES="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGES="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

# Download files
echo "Downloading training images..."
wget -O train-images-idx3-ubyte.gz "$TRAIN_IMAGES"

echo "Downloading training labels..."
wget -O train-labels-idx1-ubyte.gz "$TRAIN_LABELS"

echo "Downloading test images..."
wget -O t10k-images-idx3-ubyte.gz "$TEST_IMAGES"

echo "Downloading test labels..."
wget -O t10k-labels-idx1-ubyte.gz "$TEST_LABELS"

# Extract files
echo "Extracting files..."
gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

# Verify files
echo "Verifying downloaded files..."
if [ -f "train-images-idx3-ubyte" ] && [ -f "train-labels-idx1-ubyte" ] && \
   [ -f "t10k-images-idx3-ubyte" ] && [ -f "t10k-labels-idx1-ubyte" ]; then
    echo "MNIST dataset downloaded successfully!"
    echo "Files:"
    ls -la *.ubyte
else
    echo "Error: Some files are missing!"
    exit 1
fi

cd ../..

echo "MNIST dataset is ready for use!"
echo "You can now run: ./lvq_train --dataset data/mnist" 