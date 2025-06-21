# Learning Vector Quantization (LVQ) Network - C++ Implementation

A generic Learning Vector Quantization (LVQ) neural network implementation in C++ for classification tasks. 

## Project Structure

```
learning-vector-quantization/
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # This file
├── build.sh                    # Build script
├── include/                    # Header files
│   ├── lvq_network.h          # Main LVQ network class
│   ├── lvq_trainer.h          # Training module
│   ├── lvq_tester.h           # Testing and evaluation module
│   ├── data_loader.h          # Data loading and preprocessing
│   └── utils.h                # Utility functions and helpers
├── src/                       # Source files
│   ├── lvq_network.cpp        # LVQ network implementation
│   ├── lvq_trainer.cpp        # Training implementation
│   ├── lvq_tester.cpp         # Testing implementation
│   ├── data_loader.cpp        # Data loading implementation
│   ├── utils.cpp              # Utility implementations
│   ├── main_train.cpp         # Training executable
│   ├── main_test.cpp          # Testing executable
│   ├── main_demo.cpp          # Demo executable
│   └── main_iris_demo.cpp     # Iris dataset demo executable
├── data/                      # Dataset directory
│   ├── iris.csv               # Real Iris dataset (150 samples)
│   └── mnist/                 # MNIST dataset files
├── models/                    # Trained models (created after training)
├── results/                   # Test results (created after testing)
└── examples/                  # Example configurations and scripts
    ├── config.txt             # Example configuration file
    ├── download_mnist.sh      # MNIST download script
    └── iris_data.csv          # Example Iris dataset
```

## Building the Project

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.10 or higher
- Make or Ninja build system

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd learning-vector-quantization

# Build using the provided script
./build.sh

# Or build manually
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Quick Start - Iris Dataset Demo

The easiest way to get started is with the Iris dataset demo:

```bash
# Build the project
./build.sh

# Train and test on Iris dataset with interactive mode
./build/lvq_iris_demo --train --interactive --epochs 100 --lr 0.1 --codebook 3
```

This will:
1. Download the real Iris dataset (150 samples)
2. Train an LVQ network with 3 codebook vectors per class
3. Achieve ~100% accuracy on the test set
4. Enter interactive mode for real-time predictions

Example interactive session:
```
Enter 4 feature values (sepal length, sepal width, petal length, petal width):
5.1 3.5 1.4 0.2
Prediction: setosa (confidence: 0.160)
Top-3 predictions: setosa (0.160) versicolor (0.156) virginica (0.154)
Input features: Sepal length=5.1cm, Sepal width=3.5cm, Petal length=1.4cm, Petal width=0.2cm
```

## Usage

### 1. Training a Model

```bash
# Basic training with default parameters
./build/lvq_train

# Training with custom parameters
./build/lvq_train --dataset data/mnist --epochs 200 --lr 0.01 --codebook 15 --output my_models

# Training with configuration file
./build/lvq_train --config config.txt --output models
```

### 2. Testing a Model

```bash
# Basic testing
./build/lvq_test --model models/lvq_model.bin

# Testing with detailed analysis
./build/lvq_test --model models/lvq_model.bin --detailed --output results

# Testing on custom dataset
./build/lvq_test --model models/lvq_model.bin --dataset data/custom_dataset
```

### 3. Using the Demo Tool

```bash
# Interactive demo with trained model
./build/lvq_demo --model models/lvq_model.bin

# Batch prediction on file
./build/lvq_demo --model models/lvq_model.bin --input test_data.csv --output predictions.csv

# Demo with confidence scores and top-k predictions
./build/lvq_demo --model models/lvq_model.bin --confidence --top-k 3
```

### 4. Iris Dataset Demo

```bash
# Train and test on Iris dataset (recommended for beginners)
./build/lvq_iris_demo --train --interactive

# Train with custom parameters
./build/lvq_iris_demo --train --epochs 150 --lr 0.02 --codebook 8

# Load existing model and enter interactive mode
./build/lvq_iris_demo --model models/iris_model.bin --interactive

# Just train without interactive mode
./build/lvq_iris_demo --train --epochs 100 --lr 0.1 --codebook 3
```

## Dataset Support

### Iris Dataset (Recommended for Testing)

The project automatically downloads and uses the real Iris dataset (150 samples):

**Features:**
- Sepal length (cm): 4.3 - 7.9
- Sepal width (cm): 2.0 - 4.4  
- Petal length (cm): 1.0 - 6.9
- Petal width (cm): 0.1 - 2.5

**Classes:**
- Setosa (class 0)
- Versicolor (class 1)
- Virginica (class 2)

**Expected Performance:**
- Training time: ~0.001-0.005 seconds
- Test accuracy: ~100%
- Perfect confusion matrix

### MNIST Dataset

The project includes built-in support for the MNIST handwritten digit dataset:

1. Download MNIST files:
   ```bash
   ./examples/download_mnist.sh
   ```

2. Run training:
   ```bash
   ./build/lvq_train --dataset data/mnist
   ```

### Custom Datasets

For custom datasets, use CSV format:
```csv
feature1,feature2,feature3,...,label
0.1,0.2,0.3,...,0
0.4,0.5,0.6,...,1
...
```

Load with:
```bash
./build/lvq_train --dataset data/custom_dataset.csv
```

## Configuration

Create a configuration file `config.txt`:
```ini
learning_rate=0.01
num_codebook_vectors=10
max_iterations=1000
convergence_threshold=1e-6
use_adaptive_lr=true
distance_metric=euclidean
random_seed=42
```

## API Usage

### Basic Usage

```cpp
#include "lvq_network.h"
#include "lvq_trainer.h"
#include "lvq_tester.h"

// Create network
LVQConfig config;
config.learning_rate = 0.1;
config.num_codebook_vectors = 3;
config.max_iterations = 100;

auto network = std::make_shared<LVQNetwork>(config);

// Train
LVQTrainer trainer(network, config);
auto stats = trainer.train(training_data);

// Test
LVQTester tester(network);
auto results = tester.test(test_data);

// Predict
auto prediction = network->predict(features);
```

## Python Bindings

The project includes Python bindings that provide a high-level interface to the LVQ network with full NumPy integration.

### Installation

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Build Python module
./build_python.sh

# Activate the virtual environment and set up Python path
source activate_venv.sh
```

### Quick Start

```python
import numpy as np
import lvq_python as lvq

# Create sample data
X = np.random.randn(100, 4)  # 100 samples, 4 features
y = np.random.randint(0, 3, 100)  # 3 classes

# Convert to LVQ format
data_points = lvq.array_to_data_points(X, y)

# Configure and create network
config = lvq.LVQConfig()
config.num_codebook_vectors = 3
config.learning_rate = 0.1
config.max_iterations = 100

network = lvq.LVQNetwork(config)

# Train the network
network.train(data_points)

# Make predictions
predictions = network.predict(X[0])  # Single prediction
batch_predictions = network.predict_batch(X)  # Batch prediction
prediction, confidence = network.predict_with_confidence(X[0])  # With confidence

print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
```

### Example Usage

```python
# Basic classification
import numpy as np
import lvq_python as lvq

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(300, 4)
y = np.random.randint(0, 3, 300)

# Train network
data_points = lvq.array_to_data_points(X, y)
network = lvq.LVQNetwork()
network.train(data_points)

# Evaluate
predictions = network.predict_batch([dp.features for dp in data_points])
accuracy = np.mean(np.array(predictions) == y)
print(f"Accuracy: {accuracy:.3f}")

# Save and load model
network.save_model("my_model.bin")
new_network = lvq.LVQNetwork()
new_network.load_model("my_model.bin")
```

For detailed documentation, see [python_bindings/README.md](python_bindings/README.md).

## Performance Results

### Iris Dataset Results
- **Dataset**: 150 samples (120 train, 15 validation, 15 test)
- **Training Time**: ~0.001-0.005 seconds
- **Test Accuracy**: 100%
- **Codebook Vectors**: 9 total (3 per class)
- **Distance Metric**: Euclidean

### MNIST Dataset Results
- **Dataset**: 70,000 samples (60,000 train, 10,000 test)
- **Training Time**: ~30-60 seconds
- **Test Accuracy**: ~95-98%
- **Codebook Vectors**: 100-500 total (10-50 per class)

## Troubleshooting

### Common Issues

1. **Build Errors**: Ensure you have C++17 support
2. **Dataset Not Found**: Check that data files are in the correct location
3. **Low Accuracy**: Try adjusting learning rate or number of codebook vectors
4. **Memory Issues**: Reduce batch size or number of codebook vectors

## Acknowledgments

- Fisher's Iris dataset for the classic classification example
- MNIST dataset for computer vision testing

## References

- Kohonen, T. (1990). "The self-organizing map". Proceedings of the IEEE, 78(9), 1464-1480.
- Kohonen, T. (1995). "Learning Vector Quantization". In M. A. Arbib (Ed.), The Handbook of Brain Theory and Neural Networks (pp. 537-540). MIT Press.
- Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems". Annals of Eugenics, 7(2), 179-188.
