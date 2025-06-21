#!/bin/bash

# Script to activate the virtual environment and set up the Python path for LVQ development

echo "Activating LVQ Python virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv --without-pip
    
    # Install pip if not present
    if [ ! -f "venv/bin/pip" ]; then
        echo "Installing pip..."
        source venv/bin/activate
        curl https://bootstrap.pypa.io/get-pip.py | python3
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if not already installed
if ! python -c "import pybind11" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Find the directory containing lvq_python*.so
LVQ_SO_PATH=""
if compgen -G "lvq_python*.so" > /dev/null; then
    LVQ_SO_PATH="$(pwd)"
elif compgen -G "build/lib.*/*lvq_python*.so" > /dev/null; then
    LVQ_SO_PATH="$(pwd)/build/lib.*"
fi

if [ -n "$LVQ_SO_PATH" ]; then
    export PYTHONPATH="$PYTHONPATH:$LVQ_SO_PATH"
    echo "PYTHONPATH set to include: $LVQ_SO_PATH"
else
    export PYTHONPATH="$PYTHONPATH:$(pwd)/build"
    echo "PYTHONPATH set to include: $(pwd)/build"
fi

echo "Virtual environment activated!"
echo "To deactivate, run: deactivate"
echo "To build the Python module, run: ./build_python.sh"
echo "To run the example, run: python examples/python_example.py" 