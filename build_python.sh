#!/bin/bash

# Build script for LVQ Python bindings

set -e  # Exit on any error

echo "Building LVQ Python bindings..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed or not in PATH"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Create build directory
mkdir -p build

# Build using setup.py
echo "Building Python module..."
python3 setup.py build_ext --inplace

# Move the built module to build directory
if [ -f "lvq_python.*.so" ]; then
    mv lvq_python.*.so build/
    echo "Python module built successfully: build/lvq_python.*.so"
elif [ -f "lvq_python.pyd" ]; then
    mv lvq_python.pyd build/
    echo "Python module built successfully: build/lvq_python.pyd"
else
    echo "Warning: Could not find built Python module"
fi

# Test the build
echo "Testing Python module..."
cd build
python3 -c "import lvq_python; print('Python module imported successfully!')"
cd ..

echo "Build completed successfully!"
echo ""
echo "To use the Python bindings:"
echo "1. Add the build directory to your Python path:"
echo "   export PYTHONPATH=\$PYTHONPATH:$(pwd)/build"
echo ""
echo "2. Or install the module:"
echo "   pip3 install -e ."
echo ""
echo "3. Run the example:"
echo "   python3 examples/python_example.py" 