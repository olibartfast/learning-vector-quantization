#pragma once

#include "lvq_network.h"
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace lvq {

/**
 * @brief Classification metrics and evaluation results
 */
struct ClassificationMetrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double confusion_matrix[10][10] = {{0}}; // Support up to 10 classes
    std::map<int, double> per_class_accuracy;
    std::map<int, double> per_class_precision;
    std::map<int, double> per_class_recall;
    std::map<int, double> per_class_f1;
    
    void calculate_metrics(const std::vector<int>& predictions, 
                          const std::vector<int>& true_labels);
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void print_summary() const;
    void print_confusion_matrix() const;
};

/**
 * @brief Test results with detailed information
 */
struct TestResults {
    ClassificationMetrics metrics;
    std::vector<int> predictions;
    std::vector<double> confidence_scores;
    double testing_time_seconds = 0.0;
    int total_samples = 0;
    int correct_predictions = 0;
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void print_summary() const;
};

/**
 * @brief LVQ Tester class for handling testing and evaluation
 */
class LVQTester {
private:
    std::shared_ptr<LVQNetwork> network_;
    
    // Testing helper methods
    std::vector<double> calculate_confidence_scores(const std::vector<std::vector<double>>& features);
    void validate_test_data(const std::vector<DataPoint>& data) const;
    
public:
    // Constructors
    LVQTester() = default;
    explicit LVQTester(std::shared_ptr<LVQNetwork> network);
    
    // Destructor
    ~LVQTester() = default;
    
    // Testing methods
    TestResults test(const std::vector<DataPoint>& test_data);
    TestResults test_with_confidence(const std::vector<DataPoint>& test_data);
    
    // Batch testing
    TestResults test_batch(const std::vector<std::vector<double>>& features,
                          const std::vector<int>& labels);
    
    // Cross-validation testing
    std::vector<TestResults> cross_validate_test(const std::vector<DataPoint>& data,
                                                int num_folds = 5);
    
    // Performance analysis
    void analyze_performance(const std::vector<DataPoint>& test_data);
    void generate_error_analysis(const std::vector<DataPoint>& test_data,
                                const std::vector<int>& predictions);
    
    // Model comparison
    static std::map<std::string, TestResults> compare_models(
        const std::map<std::string, std::shared_ptr<LVQNetwork>>& models,
        const std::vector<DataPoint>& test_data);
    
    // Getters and setters
    void set_network(std::shared_ptr<LVQNetwork> network) { network_ = network; }
    std::shared_ptr<LVQNetwork> get_network() const { return network_; }
    
    // Utility methods
    bool validate_network() const;
    std::vector<std::pair<int, double>> get_top_k_predictions(
        const std::vector<double>& features, int k = 3) const;
};

} // namespace lvq
