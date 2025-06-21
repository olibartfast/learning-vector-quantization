# LVQ Network Python Bindings

This directory contains Python bindings for the Learning Vector Quantization (LVQ) Network C++ library, allowing you to use the LVQ network directly from Python with full NumPy integration.

## Features

- **Full NumPy Integration**: Work seamlessly with NumPy arrays
- **High Performance**: Direct C++ implementation with Python bindings
- **Easy to Use**: Pythonic interface with familiar ML patterns
- **Model Persistence**: Save and load trained models
- **Batch Operations**: Efficient batch prediction and training
- **Confidence Scores**: Get prediction confidence along with class predictions

## Installation

### Prerequisites

- Python 3.7 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.10 or higher (for CMake build)
- NumPy and pybind11

### Quick Installation

1. **Install Python dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Build the Python module:**
   ```bash
   ./build_python.sh
   ```

3. **Add to Python path:**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/build
   ```

### Alternative Installation Methods

**Using setup.py:**
```bash
pip3 install -e .
```

**Using CMake:**
```bash
mkdir build && cd build
cmake ../python_bindings
make
```

## Quick Start

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

## API Reference

### Core Classes

#### `LVQNetwork`

The main class for LVQ network operations.

**Constructor:**
```python
network = lvq.LVQNetwork()  # Default configuration
network = lvq.LVQNetwork(config)  # Custom configuration
```

**Methods:**
- `train(data_points)`: Train the network on a list of DataPoint objects
- `train_online(data_point)`: Train on a single data point (online learning)
- `predict(features)`: Predict class for a single sample (NumPy array or list)
- `predict_batch(features_list)`: Predict classes for multiple samples
- `predict_with_confidence(features)`: Predict with confidence score
- `save_model(filename)`: Save trained model to file
- `load_model(filename)`: Load model from file
- `reset()`: Reset the network to untrained state
- `print_summary()`: Print network information

**Properties:**
- `is_trained`: Boolean indicating if network is trained
- `get_input_dimension()`: Get input feature dimension
- `get_codebook_vectors()`: Get list of codebook vectors
- `get_unique_classes()`: Get list of unique class labels

#### `LVQConfig`

Configuration class for LVQ network parameters.

**Properties:**
- `num_codebook_vectors`: Number of prototypes per class (default: 10)
- `learning_rate`: Initial learning rate (default: 0.01)
- `learning_rate_decay`: Learning rate decay factor (default: 0.999)
- `max_iterations`: Maximum training iterations (default: 1000)
- `convergence_threshold`: Convergence threshold (default: 1e-6)
- `use_adaptive_lr`: Use adaptive learning rate (default: True)
- `distance_metric`: Distance metric ("euclidean") (default: "euclidean")
- `random_seed`: Random seed for reproducibility (default: 42)

**Methods:**
- `save(filename)`: Save configuration to file
- `load(filename)`: Load configuration from file

#### `DataPoint`

Represents a single training sample.

**Constructor:**
```python
dp = lvq.DataPoint(features, class_label)
```

**Properties:**
- `features`: List of feature values
- `class_label`: Integer class label
- `get_dimension()`: Get number of features

#### `CodebookVector`

Represents a prototype vector in the LVQ network.

**Properties:**
- `weights`: List of weight values
- `class_label`: Integer class label
- `frequency`: Usage frequency counter
- `get_dimension()`: Get number of weights

### Utility Functions

#### `array_to_data_points(features, labels)`

Convert NumPy arrays to list of DataPoint objects.

```python
# features: 2D NumPy array (n_samples, n_features)
# labels: 1D NumPy array (n_samples,)
data_points = lvq.array_to_data_points(X, y)
```

#### `data_points_to_arrays(data_points)`

Convert list of DataPoint objects to NumPy arrays.

```python
# Returns tuple (features, labels)
features, labels = lvq.data_points_to_arrays(data_points)
```

#### `create_data_point(features, class_label)`

Create a single DataPoint from NumPy array.

```python
dp = lvq.create_data_point(np.array([1.0, 2.0, 3.0]), 0)
```

## Examples

### Basic Classification

```python
import numpy as np
import lvq_python as lvq

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(300, 4)
y = np.random.randint(0, 3, 300)

# Convert to LVQ format
data_points = lvq.array_to_data_points(X, y)

# Create and train network
config = lvq.LVQConfig()
config.num_codebook_vectors = 5
config.learning_rate = 0.1
config.max_iterations = 200

network = lvq.LVQNetwork(config)
network.train(data_points)

# Evaluate
predictions = network.predict_batch([dp.features for dp in data_points])
accuracy = np.mean(np.array(predictions) == y)
print(f"Accuracy: {accuracy:.3f}")
```

### Model Persistence

```python
# Save model
network.save_model("my_model.bin")

# Load model
new_network = lvq.LVQNetwork()
new_network.load_model("my_model.bin")

# Verify predictions match
test_sample = np.array([1.0, 2.0, 3.0, 4.0])
pred1 = network.predict(test_sample)
pred2 = new_network.predict(test_sample)
print(f"Predictions match: {pred1 == pred2}")
```

### Online Learning

```python
# Train online with individual samples
for i in range(len(data_points)):
    network.train_online(data_points[i])
    
    # Make prediction after each sample
    if i % 50 == 0:
        prediction = network.predict(data_points[i].features)
        print(f"Sample {i}: predicted {prediction}, actual {data_points[i].class_label}")
```

### Confidence-based Predictions

```python
# Get predictions with confidence scores
for i in range(5):
    prediction, confidence = network.predict_with_confidence(X[i])
    print(f"Sample {i}: class {prediction}, confidence {confidence:.3f}")
```

## Performance Tips

1. **Batch Operations**: Use `predict_batch()` for multiple predictions
2. **NumPy Arrays**: Pass NumPy arrays directly for best performance
3. **Model Reuse**: Save and load models instead of retraining
4. **Appropriate Config**: Adjust `num_codebook_vectors` based on dataset size

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure the build directory is in your Python path
2. **Compilation Error**: Ensure you have a C++17 compatible compiler
3. **Memory Error**: Reduce `num_codebook_vectors` for large datasets
4. **Poor Performance**: Try adjusting learning rate and number of iterations

### Debug Information

```python
# Print network information
network.print_summary()

# Check configuration
print(network.get_config())

# Inspect codebook vectors
for i, cv in enumerate(network.get_codebook_vectors()):
    print(f"Vector {i}: class {cv.class_label}, frequency {cv.frequency}")
```

## Integration with Other Libraries

### Scikit-learn Compatibility

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import lvq_python as lvq

class LVQClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_codebook_vectors=10, learning_rate=0.01, max_iterations=1000):
        self.num_codebook_vectors = num_codebook_vectors
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.network_ = None
        
    def fit(self, X, y):
        config = lvq.LVQConfig()
        config.num_codebook_vectors = self.num_codebook_vectors
        config.learning_rate = self.learning_rate
        config.max_iterations = self.max_iterations
        
        self.network_ = lvq.LVQNetwork(config)
        data_points = lvq.array_to_data_points(X, y)
        self.network_.train(data_points)
        return self
        
    def predict(self, X):
        if self.network_ is None:
            raise ValueError("Model not fitted")
        return np.array(self.network_.predict_batch(X))
    
    def predict_proba(self, X):
        # Note: LVQ doesn't provide probabilities, but we can return confidence scores
        if self.network_ is None:
            raise ValueError("Model not fitted")
        predictions = []
        confidences = []
        for sample in X:
            pred, conf = self.network_.predict_with_confidence(sample)
            predictions.append(pred)
            confidences.append(conf)
        return np.array(confidences).reshape(-1, 1)
```

### Pandas Integration

```python
import pandas as pd
import numpy as np
import lvq_python as lvq

# Load data with pandas
df = pd.read_csv("data.csv")
X = df.drop('target', axis=1).values
y = df['target'].values

# Train LVQ network
data_points = lvq.array_to_data_points(X, y)
network = lvq.LVQNetwork()
network.train(data_points)

# Make predictions on new data
new_df = pd.read_csv("new_data.csv")
new_X = new_df.values
predictions = network.predict_batch(new_X)

# Add predictions to dataframe
new_df['predicted_class'] = predictions
```

## License

This Python binding is part of the LVQ Network project and follows the same license as the main C++ library. 