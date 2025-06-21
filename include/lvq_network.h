#pragma once


#include <vector>
#include <memory>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace lvq {

// Forward declarations
class DataPoint;
class CodebookVector;

/**
 * @brief Represents a data point with features and class label
 */
class DataPoint {
public:
    std::vector<double> features;
    int class_label;
    
    DataPoint() = default;
    DataPoint(const std::vector<double>& feat, int label) 
        : features(feat), class_label(label) {}
    
    size_t get_dimension() const { return features.size(); }
};

/**
 * @brief Represents a codebook vector (prototype) in the LVQ network
 */
class CodebookVector {
public:
    std::vector<double> weights;
    int class_label;
    int frequency;
    
    CodebookVector() = default;
    CodebookVector(const std::vector<double>& w, int label) 
        : weights(w), class_label(label), frequency(1) {}
    
    size_t get_dimension() const { return weights.size(); }
};

/**
 * @brief Configuration parameters for LVQ network
 */
struct LVQConfig {
    int num_codebook_vectors = 10;  // Number of prototypes per class
    double learning_rate = 0.01;    // Initial learning rate
    double learning_rate_decay = 0.999; // Learning rate decay factor
    int max_iterations = 1000;      // Maximum training iterations
    double convergence_threshold = 1e-6; // Convergence threshold
    bool use_adaptive_lr = true;    // Use adaptive learning rate
    std::string distance_metric = "euclidean"; // Distance metric
    int random_seed = 42;           // Random seed for reproducibility
    
    // Save/load configuration
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};

/**
 * @brief Main LVQ Network class for production use
 */
class LVQNetwork {
private:
    std::vector<CodebookVector> codebook_vectors_;
    LVQConfig config_;
    std::mt19937 rng_;
    bool is_trained_;
    std::vector<int> unique_classes_;
    
    // Private helper methods
    double calculate_distance(const std::vector<double>& v1, 
                             const std::vector<double>& v2) const;
    int find_winner(const std::vector<double>& input) const;
    void update_codebook_vector(CodebookVector& codebook, 
                               const DataPoint& data_point, 
                               double learning_rate, 
                               bool same_class);
    void initialize_codebook_vectors(const std::vector<DataPoint>& training_data);
    std::vector<double> normalize_features(const std::vector<double>& features) const;
    
public:
    // Constructors
    LVQNetwork() = default;
    explicit LVQNetwork(const LVQConfig& config);
    
    // Destructor
    ~LVQNetwork() = default;
    
    // Copy and move constructors/assignments
    LVQNetwork(const LVQNetwork& other) = default;
    LVQNetwork(LVQNetwork&& other) = default;
    LVQNetwork& operator=(const LVQNetwork& other) = default;
    LVQNetwork& operator=(LVQNetwork&& other) = default;
    
    // Training methods
    void train(const std::vector<DataPoint>& training_data);
    void train_online(const DataPoint& data_point);
    
    // Prediction methods
    int predict(const std::vector<double>& features) const;
    std::vector<int> predict_batch(const std::vector<std::vector<double>>& features) const;
    std::pair<int, double> predict_with_confidence(const std::vector<double>& features) const;
    
    // Model persistence
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Getters
    const std::vector<CodebookVector>& get_codebook_vectors() const { return codebook_vectors_; }
    const LVQConfig& get_config() const { return config_; }
    bool is_trained() const { return is_trained_; }
    std::vector<int> get_unique_classes() const { return unique_classes_; }
    size_t get_input_dimension() const;
    
    // Setters
    void set_config(const LVQConfig& config);
    
    // Utility methods
    void reset();
    void print_summary() const;
    std::vector<std::vector<double>> get_decision_boundaries() const;
};

} // namespace lvq
