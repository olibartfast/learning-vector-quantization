#pragma once

#include "lvq_network.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace lvq {

/**
 * @brief Training statistics and metrics
 */
struct TrainingStats {
    double initial_error = 0.0;
    double final_error = 0.0;
    double convergence_error = 0.0;
    int iterations_completed = 0;
    double training_time_seconds = 0.0;
    std::vector<double> error_history;
    std::vector<double> learning_rate_history;
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void print_summary() const;
};

/**
 * @brief Training callbacks for monitoring progress
 */
struct TrainingCallbacks {
    std::function<void(int iteration, double error, double lr)> on_iteration;
    std::function<void(const TrainingStats&)> on_complete;
    std::function<bool(int iteration, double error)> should_stop_early;
    
    TrainingCallbacks() = default;
};

/**
 * @brief LVQ Trainer class for handling training operations
 */
class LVQTrainer {
private:
    std::shared_ptr<LVQNetwork> network_;
    LVQConfig config_;
    TrainingCallbacks callbacks_;
    
    // Training helper methods
    double calculate_training_error(const std::vector<DataPoint>& training_data);
    bool check_convergence(const std::vector<double>& error_history);
    void update_learning_rate(double& current_lr, int iteration);
    std::vector<DataPoint> shuffle_training_data(const std::vector<DataPoint>& data);
    
public:
    // Constructors
    LVQTrainer() = default;
    explicit LVQTrainer(std::shared_ptr<LVQNetwork> network);
    LVQTrainer(std::shared_ptr<LVQNetwork> network, const LVQConfig& config);
    
    // Destructor
    ~LVQTrainer() = default;
    
    // Training methods
    TrainingStats train(const std::vector<DataPoint>& training_data);
    TrainingStats train_with_validation(const std::vector<DataPoint>& training_data,
                                       const std::vector<DataPoint>& validation_data);
    TrainingStats train_online(const std::vector<DataPoint>& training_data);
    
    // Cross-validation
    std::vector<TrainingStats> cross_validate(const std::vector<DataPoint>& data, 
                                             int num_folds = 5);
    
    // Hyperparameter optimization
    LVQConfig optimize_hyperparameters(const std::vector<DataPoint>& training_data,
                                      const std::vector<DataPoint>& validation_data,
                                      const std::vector<LVQConfig>& configs);
    
    // Getters and setters
    void set_callbacks(const TrainingCallbacks& callbacks) { callbacks_ = callbacks; }
    const TrainingCallbacks& get_callbacks() const { return callbacks_; }
    void set_config(const LVQConfig& config) { config_ = config; }
    const LVQConfig& get_config() const { return config_; }
    std::shared_ptr<LVQNetwork> get_network() const { return network_; }
    
    // Utility methods
    void reset();
    bool validate_training_data(const std::vector<DataPoint>& data) const;
    std::vector<DataPoint> create_mini_batch(const std::vector<DataPoint>& data,
                                            size_t batch_size,
                                            size_t start_idx);
};

} // namespace lvq
