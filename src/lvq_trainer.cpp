#include "lvq_trainer.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <set>

namespace lvq {

// TrainingStats implementation
void TrainingStats::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file << "initial_error=" << initial_error << std::endl;
    file << "final_error=" << final_error << std::endl;
    file << "convergence_error=" << convergence_error << std::endl;
    file << "iterations_completed=" << iterations_completed << std::endl;
    file << "training_time_seconds=" << training_time_seconds << std::endl;
    
    // Save error history
    file << "error_history=";
    for (size_t i = 0; i < error_history.size(); i++) {
        file << error_history[i];
        if (i < error_history.size() - 1) file << ",";
    }
    file << std::endl;
    
    // Save learning rate history
    file << "learning_rate_history=";
    for (size_t i = 0; i < learning_rate_history.size(); i++) {
        file << learning_rate_history[i];
        if (i < learning_rate_history.size() - 1) file << ",";
    }
    file << std::endl;
}

void TrainingStats::load(const std::string& filename) {
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
            
            if (key == "initial_error") initial_error = std::stod(value);
            else if (key == "final_error") final_error = std::stod(value);
            else if (key == "convergence_error") convergence_error = std::stod(value);
            else if (key == "iterations_completed") iterations_completed = std::stoi(value);
            else if (key == "training_time_seconds") training_time_seconds = std::stod(value);
            else if (key == "error_history") {
                error_history.clear();
                std::istringstream iss(value);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    error_history.push_back(std::stod(token));
                }
            }
            else if (key == "learning_rate_history") {
                learning_rate_history.clear();
                std::istringstream iss(value);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    learning_rate_history.push_back(std::stod(token));
                }
            }
        }
    }
}

void TrainingStats::print_summary() const {
    std::cout << "Training Statistics:" << std::endl;
    std::cout << "  Initial error: " << initial_error << std::endl;
    std::cout << "  Final error: " << final_error << std::endl;
    std::cout << "  Convergence error: " << convergence_error << std::endl;
    std::cout << "  Iterations completed: " << iterations_completed << std::endl;
    std::cout << "  Training time: " << training_time_seconds << " seconds" << std::endl;
}

// LVQTrainer implementation
LVQTrainer::LVQTrainer(std::shared_ptr<LVQNetwork> network) : network_(network) {}

LVQTrainer::LVQTrainer(std::shared_ptr<LVQNetwork> network, const LVQConfig& config) 
    : network_(network), config_(config) {}

double LVQTrainer::calculate_training_error(const std::vector<DataPoint>& training_data) {
    if (!network_ || !network_->is_trained()) {
        return std::numeric_limits<double>::max();
    }
    
    double total_error = 0.0;
    for (const auto& data_point : training_data) {
        auto prediction = network_->predict_with_confidence(data_point.features);
        if (prediction.first != data_point.class_label) {
            total_error += 1.0; // Classification error
        }
    }
    
    return total_error / training_data.size();
}

bool LVQTrainer::check_convergence(const std::vector<double>& error_history) {
    if (error_history.size() < 10) return false;
    
    // Check if error has stabilized (last 10 iterations)
    double recent_avg = 0.0;
    for (size_t i = error_history.size() - 10; i < error_history.size(); i++) {
        recent_avg += error_history[i];
    }
    recent_avg /= 10.0;
    
    // Check if recent average is close to the last error
    double last_error = error_history.back();
    return std::abs(recent_avg - last_error) < config_.convergence_threshold;
}

void LVQTrainer::update_learning_rate(double& current_lr, int iteration) {
    if (config_.use_adaptive_lr) {
        current_lr = config_.learning_rate * std::pow(config_.learning_rate_decay, iteration);
    }
}

std::vector<DataPoint> LVQTrainer::shuffle_training_data(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> shuffled = data;
    std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(config_.random_seed));
    return shuffled;
}

TrainingStats LVQTrainer::train(const std::vector<DataPoint>& training_data) {
    if (!network_) {
        throw std::runtime_error("No network assigned to trainer");
    }
    
    if (!validate_training_data(training_data)) {
        throw std::invalid_argument("Invalid training data");
    }
    
    Timer timer;
    timer.start();
    
    TrainingStats stats;
    stats.initial_error = calculate_training_error(training_data);
    
    // Train the network
    network_->train(training_data);
    
    timer.stop();
    stats.training_time_seconds = timer.elapsed_seconds();
    stats.final_error = calculate_training_error(training_data);
    stats.convergence_error = stats.final_error;
    stats.iterations_completed = config_.max_iterations;
    
    // Call completion callback if set
    if (callbacks_.on_complete) {
        callbacks_.on_complete(stats);
    }
    
    return stats;
}

TrainingStats LVQTrainer::train_with_validation(const std::vector<DataPoint>& training_data,
                                               const std::vector<DataPoint>& validation_data) {
    if (!network_) {
        throw std::runtime_error("No network assigned to trainer");
    }
    
    if (!validate_training_data(training_data) || !validate_training_data(validation_data)) {
        throw std::invalid_argument("Invalid training or validation data");
    }
    
    Timer timer;
    timer.start();
    
    TrainingStats stats;
    stats.initial_error = calculate_training_error(training_data);
    
    // Train the network
    network_->train(training_data);
    
    timer.stop();
    stats.training_time_seconds = timer.elapsed_seconds();
    stats.final_error = calculate_training_error(validation_data);
    stats.convergence_error = stats.final_error;
    stats.iterations_completed = config_.max_iterations;
    
    // Call completion callback if set
    if (callbacks_.on_complete) {
        callbacks_.on_complete(stats);
    }
    
    return stats;
}

TrainingStats LVQTrainer::train_online(const std::vector<DataPoint>& training_data) {
    if (!network_) {
        throw std::runtime_error("No network assigned to trainer");
    }
    
    if (!validate_training_data(training_data)) {
        throw std::invalid_argument("Invalid training data");
    }
    
    Timer timer;
    timer.start();
    
    TrainingStats stats;
    stats.initial_error = calculate_training_error(training_data);
    
    // Online training - process each data point once
    for (const auto& data_point : training_data) {
        network_->train_online(data_point);
    }
    
    timer.stop();
    stats.training_time_seconds = timer.elapsed_seconds();
    stats.final_error = calculate_training_error(training_data);
    stats.convergence_error = stats.final_error;
    stats.iterations_completed = 1; // Online training is single pass
    
    // Call completion callback if set
    if (callbacks_.on_complete) {
        callbacks_.on_complete(stats);
    }
    
    return stats;
}

std::vector<TrainingStats> LVQTrainer::cross_validate(const std::vector<DataPoint>& data, 
                                                     int num_folds) {
    if (num_folds < 2 || num_folds > static_cast<int>(data.size())) {
        throw std::invalid_argument("Invalid number of folds");
    }
    
    std::vector<TrainingStats> results;
    std::vector<DataPoint> shuffled_data = shuffle_training_data(data);
    
    int fold_size = data.size() / num_folds;
    
    for (int fold = 0; fold < num_folds; fold++) {
        // Split data into training and validation
        std::vector<DataPoint> validation_data;
        std::vector<DataPoint> training_data;
        
        int start_idx = fold * fold_size;
        int end_idx = (fold == num_folds - 1) ? data.size() : (fold + 1) * fold_size;
        
        for (size_t i = 0; i < data.size(); i++) {
            if (i >= start_idx && i < end_idx) {
                validation_data.push_back(shuffled_data[i]);
            } else {
                training_data.push_back(shuffled_data[i]);
            }
        }
        
        // Create a new network for this fold
        auto fold_network = std::make_shared<LVQNetwork>(config_);
        LVQTrainer fold_trainer(fold_network, config_);
        
        // Train and evaluate
        auto fold_stats = fold_trainer.train_with_validation(training_data, validation_data);
        results.push_back(fold_stats);
    }
    
    return results;
}

LVQConfig LVQTrainer::optimize_hyperparameters(const std::vector<DataPoint>& training_data,
                                              const std::vector<DataPoint>& validation_data,
                                              const std::vector<LVQConfig>& configs) {
    if (configs.empty()) {
        throw std::invalid_argument("No configurations provided for optimization");
    }
    
    LVQConfig best_config = configs[0];
    double best_score = std::numeric_limits<double>::max();
    
    for (const auto& config : configs) {
        // Create network with this configuration
        auto test_network = std::make_shared<LVQNetwork>(config);
        LVQTrainer test_trainer(test_network, config);
        
        // Train and evaluate
        auto stats = test_trainer.train_with_validation(training_data, validation_data);
        
        if (stats.final_error < best_score) {
            best_score = stats.final_error;
            best_config = config;
        }
    }
    
    return best_config;
}

void LVQTrainer::reset() {
    if (network_) {
        network_->reset();
    }
}

bool LVQTrainer::validate_training_data(const std::vector<DataPoint>& data) const {
    if (data.empty()) return false;
    
    size_t expected_dim = data[0].features.size();
    std::set<int> classes;
    
    for (const auto& data_point : data) {
        if (data_point.features.size() != expected_dim) return false;
        classes.insert(data_point.class_label);
    }
    
    return classes.size() >= 2; // At least 2 classes needed for classification
}

std::vector<DataPoint> LVQTrainer::create_mini_batch(const std::vector<DataPoint>& data,
                                                    size_t batch_size,
                                                    size_t start_idx) {
    std::vector<DataPoint> batch;
    size_t end_idx = std::min(start_idx + batch_size, data.size());
    
    for (size_t i = start_idx; i < end_idx; i++) {
        batch.push_back(data[i]);
    }
    
    return batch;
}

} // namespace lvq 