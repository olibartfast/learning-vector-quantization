#!/usr/bin/env python3
"""
Example script demonstrating the use of LVQ Python bindings
"""

import numpy as np
import lvq_python as lvq

def main():
    print("LVQ Network Python Bindings Example")
    print("=" * 40)
    
    # Create some sample data (Iris-like dataset)
    np.random.seed(42)
    
    # Generate synthetic data with 3 classes
    n_samples_per_class = 50
    n_features = 4
    
    # Class 0: centered around (0, 0, 0, 0)
    class_0 = np.random.normal(0, 1, (n_samples_per_class, n_features))
    
    # Class 1: centered around (2, 2, 2, 2)
    class_1 = np.random.normal(2, 1, (n_samples_per_class, n_features))
    
    # Class 2: centered around (4, 4, 4, 4)
    class_2 = np.random.normal(4, 1, (n_samples_per_class, n_features))
    
    # Combine all data
    X = np.vstack([class_0, class_1, class_2])
    y = np.hstack([np.zeros(n_samples_per_class), 
                   np.ones(n_samples_per_class), 
                   np.full(n_samples_per_class, 2)])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Convert to LVQ DataPoint objects
    data_points = lvq.array_to_data_points(X, y.astype(int))
    print(f"Created {len(data_points)} data points")
    
    # Create LVQ configuration
    config = lvq.LVQConfig()
    config.num_codebook_vectors = 3  # 3 prototypes per class
    config.learning_rate = 0.1
    config.max_iterations = 100
    config.random_seed = 42
    
    print(f"\nConfiguration:")
    print(f"  Codebook vectors per class: {config.num_codebook_vectors}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max iterations: {config.max_iterations}")
    
    # Create and train the LVQ network
    print("\nTraining LVQ network...")
    network = lvq.LVQNetwork(config)
    
    # Train the network
    network.train(data_points)
    
    print(f"Training completed!")
    print(f"Network trained: {network.is_trained()}")
    print(f"Input dimension: {network.get_input_dimension()}")
    print(f"Number of codebook vectors: {len(network.get_codebook_vectors())}")
    
    # Print codebook vectors
    print("\nCodebook vectors:")
    for i, cv in enumerate(network.get_codebook_vectors()):
        print(f"  Vector {i}: class {cv.class_label}, frequency {cv.frequency}")
        print(f"    Weights: {cv.weights[:3]}...")  # Show first 3 weights
    
    # Test predictions
    print("\nTesting predictions...")
    
    # Test on training data
    predictions = network.predict_batch([dp.features for dp in data_points])
    accuracy = np.mean(np.array(predictions) == y)
    print(f"Training accuracy: {accuracy:.3f}")
    
    # Test individual predictions
    test_samples = [
        [0.1, 0.2, 0.3, 0.4],  # Should be class 0
        [2.1, 2.2, 2.3, 2.4],  # Should be class 1
        [4.1, 4.2, 4.3, 4.4],  # Should be class 2
    ]
    
    print("\nIndividual predictions:")
    for i, sample in enumerate(test_samples):
        prediction, confidence = network.predict_with_confidence(sample)
        print(f"  Sample {i+1}: {sample}")
        print(f"    Prediction: class {prediction}, confidence: {confidence:.3f}")
    
    # Test with numpy arrays
    print("\nTesting with numpy arrays...")
    test_array = np.array(test_samples)
    for i in range(len(test_array)):
        prediction = network.predict(test_array[i])
        print(f"  Sample {i+1} (numpy): class {prediction}")
    
    # Save and load model
    print("\nTesting model persistence...")
    model_filename = "test_model.bin"
    network.save_model(model_filename)
    print(f"Model saved to {model_filename}")
    
    # Load model in a new network
    new_network = lvq.LVQNetwork()
    new_network.load_model(model_filename)
    print(f"Model loaded, trained: {new_network.is_trained()}")
    
    # Test that predictions are the same
    test_pred_original = network.predict(test_samples[0])
    test_pred_loaded = new_network.predict(test_samples[0])
    print(f"Predictions match: {test_pred_original == test_pred_loaded}")
    
    # Print network summary
    print("\nNetwork summary:")
    network.print_summary()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 