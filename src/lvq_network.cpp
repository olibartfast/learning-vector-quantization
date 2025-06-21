#include "lvq_network.h"
#include "utils.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <set>

namespace lvq {

// LVQConfig implementation
void LVQConfig::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file << "num_codebook_vectors=" << num_codebook_vectors << std::endl;
    file << "learning_rate=" << learning_rate << std::endl;
    file << "learning_rate_decay=" << learning_rate_decay << std::endl;
    file << "max_iterations=" << max_iterations << std::endl;
    file << "convergence_threshold=" << convergence_threshold << std::endl;
    file << "use_adaptive_lr=" << (use_adaptive_lr ? "true" : "false") << std::endl;
    file << "distance_metric=" << distance_metric << std::endl;
    file << "random_seed=" << random_seed << std::endl;
}

void LVQConfig::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            if (key == "num_codebook_vectors") num_codebook_vectors = std::stoi(value);
            else if (key == "learning_rate") learning_rate = std::stod(value);
            else if (key == "learning_rate_decay") learning_rate_decay = std::stod(value);
            else if (key == "max_iterations") max_iterations = std::stoi(value);
            else if (key == "convergence_threshold") convergence_threshold = std::stod(value);
            else if (key == "use_adaptive_lr") use_adaptive_lr = (value == "true");
            else if (key == "distance_metric") distance_metric = value;
            else if (key == "random_seed") random_seed = std::stoi(value);
        }
    }
}

// LVQNetwork implementation
LVQNetwork::LVQNetwork(const LVQConfig& config) : config_(config) {
    rng_.seed(config.random_seed);
}

void LVQNetwork::set_config(const LVQConfig& config) {
    config_ = config;
    rng_.seed(config.random_seed);
}

double LVQNetwork::calculate_distance(const std::vector<double>& v1, 
                                     const std::vector<double>& v2) const {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }
    
    if (config_.distance_metric == "euclidean") {
        return math_utils::euclidean_distance(v1, v2);
    } else if (config_.distance_metric == "manhattan") {
        return math_utils::manhattan_distance(v1, v2);
    } else if (config_.distance_metric == "cosine") {
        return 1.0 - math_utils::cosine_similarity(v1, v2); // Convert similarity to distance
    } else {
        throw std::invalid_argument("Unknown distance metric: " + config_.distance_metric);
    }
}

int LVQNetwork::find_winner(const std::vector<double>& input) const {
    if (codebook_vectors_.empty()) {
        throw std::runtime_error("No codebook vectors available");
    }
    
    double min_distance = std::numeric_limits<double>::max();
    int winner_index = 0;
    
    for (size_t i = 0; i < codebook_vectors_.size(); i++) {
        double distance = calculate_distance(input, codebook_vectors_[i].weights);
        if (distance < min_distance) {
            min_distance = distance;
            winner_index = i;
        }
    }
    
    return winner_index;
}

void LVQNetwork::update_codebook_vector(CodebookVector& codebook, 
                                       const DataPoint& data_point, 
                                       double learning_rate, 
                                       bool same_class) {
    int sign = same_class ? 1 : -1;
    
    for (size_t i = 0; i < codebook.weights.size(); i++) {
        codebook.weights[i] += sign * learning_rate * 
                              (data_point.features[i] - codebook.weights[i]);
    }
    
    if (same_class) {
        codebook.frequency++;
    }
}

void LVQNetwork::initialize_codebook_vectors(const std::vector<DataPoint>& training_data) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data is empty");
    }
    
    // Get unique classes
    std::set<int> classes;
    for (const auto& data_point : training_data) {
        classes.insert(data_point.class_label);
    }
    unique_classes_ = std::vector<int>(classes.begin(), classes.end());
    
    // Calculate total number of codebook vectors needed
    int total_codebook_vectors = unique_classes_.size() * config_.num_codebook_vectors;
    codebook_vectors_.resize(total_codebook_vectors);
    
    // Initialize codebook vectors for each class
    int codebook_index = 0;
    for (int class_label : unique_classes_) {
        // Find data points for this class
        std::vector<DataPoint> class_data;
        for (const auto& data_point : training_data) {
            if (data_point.class_label == class_label) {
                class_data.push_back(data_point);
            }
        }
        
        // Initialize codebook vectors for this class
        for (int i = 0; i < config_.num_codebook_vectors; i++) {
            // Randomly select a data point from this class
            std::uniform_int_distribution<int> dist(0, class_data.size() - 1);
            int random_index = dist(rng_);
            
            codebook_vectors_[codebook_index] = CodebookVector(
                class_data[random_index].features, class_label);
            codebook_index++;
        }
    }
}

std::vector<double> LVQNetwork::normalize_features(const std::vector<double>& features) const {
    // Simple min-max normalization to [0, 1]
    auto [min_val, max_val] = math_utils::minmax(features);
    if (max_val == min_val) return features; // Avoid division by zero
    
    std::vector<double> normalized(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        normalized[i] = (features[i] - min_val) / (max_val - min_val);
    }
    return normalized;
}

void LVQNetwork::train(const std::vector<DataPoint>& training_data) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data is empty");
    }
    
    // Initialize codebook vectors
    initialize_codebook_vectors(training_data);
    
    // Training loop
    double current_lr = config_.learning_rate;
    
    for (int iteration = 0; iteration < config_.max_iterations; iteration++) {
        double total_error = 0.0;
        
        // Shuffle training data
        std::vector<DataPoint> shuffled_data = training_data;
        std::shuffle(shuffled_data.begin(), shuffled_data.end(), rng_);
        
        for (const auto& data_point : shuffled_data) {
            // Find winner
            int winner_index = find_winner(data_point.features);
            
            // Check if winner has same class
            bool same_class = (codebook_vectors_[winner_index].class_label == data_point.class_label);
            
            // Update winner
            update_codebook_vector(codebook_vectors_[winner_index], data_point, current_lr, same_class);
            
            // Calculate error
            double distance = calculate_distance(data_point.features, 
                                               codebook_vectors_[winner_index].weights);
            total_error += distance;
        }
        
        // Update learning rate
        if (config_.use_adaptive_lr) {
            current_lr *= config_.learning_rate_decay;
        }
        
        // Check convergence
        double avg_error = total_error / training_data.size();
        if (avg_error < config_.convergence_threshold) {
            break;
        }
    }
    
    is_trained_ = true;
}

void LVQNetwork::train_online(const DataPoint& data_point) {
    if (!is_trained_) {
        throw std::runtime_error("Network must be trained before online training");
    }
    
    int winner_index = find_winner(data_point.features);
    bool same_class = (codebook_vectors_[winner_index].class_label == data_point.class_label);
    
    update_codebook_vector(codebook_vectors_[winner_index], data_point, 
                          config_.learning_rate, same_class);
}

int LVQNetwork::predict(const std::vector<double>& features) const {
    if (!is_trained_) {
        throw std::runtime_error("Network is not trained");
    }
    
    int winner_index = find_winner(features);
    return codebook_vectors_[winner_index].class_label;
}

std::vector<int> LVQNetwork::predict_batch(const std::vector<std::vector<double>>& features) const {
    std::vector<int> predictions;
    predictions.reserve(features.size());
    
    for (const auto& feature : features) {
        predictions.push_back(predict(feature));
    }
    
    return predictions;
}

std::pair<int, double> LVQNetwork::predict_with_confidence(const std::vector<double>& features) const {
    if (!is_trained_) {
        throw std::runtime_error("Network is not trained");
    }
    
    int winner_index = find_winner(features);
    double distance = calculate_distance(features, codebook_vectors_[winner_index].weights);
    
    // Convert distance to confidence (inverse relationship)
    double confidence = 1.0 / (1.0 + distance);
    
    return {codebook_vectors_[winner_index].class_label, confidence};
}

void LVQNetwork::save_model(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Write configuration
    config_.save(filename + ".config");
    
    // Write codebook vectors
    size_t num_vectors = codebook_vectors_.size();
    file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
    
    for (const auto& codebook : codebook_vectors_) {
        size_t num_features = codebook.weights.size();
        file.write(reinterpret_cast<const char*>(&num_features), sizeof(num_features));
        file.write(reinterpret_cast<const char*>(codebook.weights.data()), 
                   num_features * sizeof(double));
        file.write(reinterpret_cast<const char*>(&codebook.class_label), sizeof(int));
        file.write(reinterpret_cast<const char*>(&codebook.frequency), sizeof(int));
    }
    
    // Write unique classes
    size_t num_classes = unique_classes_.size();
    file.write(reinterpret_cast<const char*>(&num_classes), sizeof(num_classes));
    file.write(reinterpret_cast<const char*>(unique_classes_.data()), 
               num_classes * sizeof(int));
    
    // Write training status
    file.write(reinterpret_cast<const char*>(&is_trained_), sizeof(bool));
}

void LVQNetwork::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    
    // Load configuration
    config_.load(filename + ".config");
    
    // Load codebook vectors
    size_t num_vectors;
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
    codebook_vectors_.resize(num_vectors);
    
    for (auto& codebook : codebook_vectors_) {
        size_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), sizeof(num_features));
        codebook.weights.resize(num_features);
        file.read(reinterpret_cast<char*>(codebook.weights.data()), 
                  num_features * sizeof(double));
        file.read(reinterpret_cast<char*>(&codebook.class_label), sizeof(int));
        file.read(reinterpret_cast<char*>(&codebook.frequency), sizeof(int));
    }
    
    // Load unique classes
    size_t num_classes;
    file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));
    unique_classes_.resize(num_classes);
    file.read(reinterpret_cast<char*>(unique_classes_.data()), 
              num_classes * sizeof(int));
    
    // Load training status
    file.read(reinterpret_cast<char*>(&is_trained_), sizeof(bool));
}

size_t LVQNetwork::get_input_dimension() const {
    if (codebook_vectors_.empty()) return 0;
    return codebook_vectors_[0].weights.size();
}

void LVQNetwork::reset() {
    codebook_vectors_.clear();
    unique_classes_.clear();
    is_trained_ = false;
}

void LVQNetwork::print_summary() const {
    std::cout << "LVQ Network Summary:" << std::endl;
    std::cout << "  Trained: " << (is_trained_ ? "Yes" : "No") << std::endl;
    std::cout << "  Input dimension: " << get_input_dimension() << std::endl;
    std::cout << "  Number of codebook vectors: " << codebook_vectors_.size() << std::endl;
    std::cout << "  Number of classes: " << unique_classes_.size() << std::endl;
    
    if (!unique_classes_.empty()) {
        std::cout << "  Classes: ";
        for (size_t i = 0; i < unique_classes_.size(); i++) {
            std::cout << unique_classes_[i];
            if (i < unique_classes_.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "  Configuration:" << std::endl;
    std::cout << "    Learning rate: " << config_.learning_rate << std::endl;
    std::cout << "    Max iterations: " << config_.max_iterations << std::endl;
    std::cout << "    Distance metric: " << config_.distance_metric << std::endl;
}

std::vector<std::vector<double>> LVQNetwork::get_decision_boundaries() const {
    // This is a simplified implementation
    // In practice, you would need more sophisticated boundary calculation
    std::vector<std::vector<double>> boundaries;
    
    for (const auto& codebook : codebook_vectors_) {
        boundaries.push_back(codebook.weights);
    }
    
    return boundaries;
}

} // namespace lvq 