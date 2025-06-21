#!/bin/bash

# LVQ Network Build Script
# This script builds the LVQ network project using CMake

set -e  # Exit on any error

echo "Building LVQ Network Project..."

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed. Please install CMake first."
    exit 1
fi

# Check if make is installed
if ! command -v make &> /dev/null; then
    echo "Error: Make is not installed. Please install Make first."
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Navigate to build directory
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executables created:"
echo "  - lvq_train: Training tool"
echo "  - lvq_test: Testing tool"
echo "  - lvq_demo: Demo tool"
echo "  - lvq_iris_demo: Iris dataset demo tool"

# Check if executables were created
if [ -f "lvq_train" ] && [ -f "lvq_test" ] && [ -f "lvq_demo" ] && [ -f "lvq_iris_demo" ]; then
    echo "All executables created successfully!"
else
    echo "Warning: Some executables may not have been created."
fi

echo ""
echo "To run the training tool:"
echo "  ./lvq_train --help"
echo ""
echo "To run the testing tool:"
echo "  ./lvq_test --help"
echo ""
echo "To run the demo tool:"
echo "  ./lvq_demo --help"
echo ""
echo "To run the Iris demo:"
echo "  ./lvq_iris_demo --help"
echo ""
echo "Example Iris demo usage:"
echo "  ./lvq_iris_demo --train --interactive" 