#include "lvq_tester.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace lvq {

// ClassificationMetrics implementation
void ClassificationMetrics::calculate_metrics(const std::vector<int>& predictions, 
                                             const std::vector<int>& true_labels) {
    if (predictions.size() != true_labels.size()) {
        throw std::invalid_argument("Predictions and true labels must have same size");
    }
    
    // Calculate confusion matrix
    std::map<int, int> class_counts;
    for (int label : true_labels) {
        class_counts[label]++;
    }
    
    int num_classes = class_counts.size();
    if (num_classes > 10) {
        throw std::runtime_error("Too many classes for confusion matrix (max 10)");
    }
    
    // Initialize confusion matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            confusion_matrix[i][j] = 0;
        }
    }
    
    // Fill confusion matrix
    for (size_t i = 0; i < predictions.size(); i++) {
        int pred = predictions[i];
        int true_label = true_labels[i];
        confusion_matrix[true_label][pred]++;
    }
    
    // Calculate overall accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == true_labels[i]) {
            correct++;
        }
    }
    accuracy = static_cast<double>(correct) / predictions.size();
    
    // Calculate per-class metrics
    for (const auto& [class_label, count] : class_counts) {
        int tp = confusion_matrix[class_label][class_label];
        int fp = 0, fn = 0;
        
        // Calculate false positives and false negatives
        for (int i = 0; i < num_classes; i++) {
            if (i != class_label) {
                fp += confusion_matrix[i][class_label];
                fn += confusion_matrix[class_label][i];
            }
        }
        
        // Per-class accuracy
        per_class_accuracy[class_label] = static_cast<double>(tp) / count;
        
        // Per-class precision
        if (tp + fp > 0) {
            per_class_precision[class_label] = static_cast<double>(tp) / (tp + fp);
        } else {
            per_class_precision[class_label] = 0.0;
        }
        
        // Per-class recall
        if (tp + fn > 0) {
            per_class_recall[class_label] = static_cast<double>(tp) / (tp + fn);
        } else {
            per_class_recall[class_label] = 0.0;
        }
        
        // Per-class F1-score
        double prec = per_class_precision[class_label];
        double rec = per_class_recall[class_label];
        if (prec + rec > 0) {
            per_class_f1[class_label] = 2.0 * (prec * rec) / (prec + rec);
        } else {
            per_class_f1[class_label] = 0.0;
        }
    }
    
    // Calculate overall precision, recall, and F1-score (macro-averaged)
    precision = 0.0;
    recall = 0.0;
    f1_score = 0.0;
    
    for (const auto& [class_label, count] : class_counts) {
        precision += per_class_precision[class_label];
        recall += per_class_recall[class_label];
        f1_score += per_class_f1[class_label];
    }
    
    precision /= num_classes;
    recall /= num_classes;
    f1_score /= num_classes;
}

void ClassificationMetrics::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file << "accuracy=" << accuracy << std::endl;
    file << "precision=" << precision << std::endl;
    file << "recall=" << recall << std::endl;
    file << "f1_score=" << f1_score << std::endl;
    
    // Save per-class metrics
    for (const auto& [class_label, acc] : per_class_accuracy) {
        file << "class_" << class_label << "_accuracy=" << acc << std::endl;
    }
    
    for (const auto& [class_label, prec] : per_class_precision) {
        file << "class_" << class_label << "_precision=" << prec << std::endl;
    }
    
    for (const auto& [class_label, rec] : per_class_recall) {
        file << "class_" << class_label << "_recall=" << rec << std::endl;
    }
    
    for (const auto& [class_label, f1] : per_class_f1) {
        file << "class_" << class_label << "_f1=" << f1 << std::endl;
    }
}

void ClassificationMetrics::load(const std::string& filename) {
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
            
            if (key == "accuracy") accuracy = std::stod(value);
            else if (key == "precision") precision = std::stod(value);
            else if (key == "recall") recall = std::stod(value);
            else if (key == "f1_score") f1_score = std::stod(value);
            else if (key.find("_accuracy=") != std::string::npos) {
                int class_label = std::stoi(key.substr(6, key.find("_accuracy=") - 6));
                per_class_accuracy[class_label] = std::stod(value);
            }
            else if (key.find("_precision=") != std::string::npos) {
                int class_label = std::stoi(key.substr(6, key.find("_precision=") - 6));
                per_class_precision[class_label] = std::stod(value);
            }
            else if (key.find("_recall=") != std::string::npos) {
                int class_label = std::stoi(key.substr(6, key.find("_recall=") - 6));
                per_class_recall[class_label] = std::stod(value);
            }
            else if (key.find("_f1=") != std::string::npos) {
                int class_label = std::stoi(key.substr(6, key.find("_f1=") - 6));
                per_class_f1[class_label] = std::stod(value);
            }
        }
    }
}

void ClassificationMetrics::print_summary() const {
    std::cout << "Classification Metrics:" << std::endl;
    std::cout << "  Overall Accuracy: " << std::fixed << std::setprecision(4) << accuracy << std::endl;
    std::cout << "  Overall Precision: " << precision << std::endl;
    std::cout << "  Overall Recall: " << recall << std::endl;
    std::cout << "  Overall F1-Score: " << f1_score << std::endl;
    
    std::cout << "  Per-class metrics:" << std::endl;
    for (const auto& [class_label, acc] : per_class_accuracy) {
        std::cout << "    Class " << class_label << ":" << std::endl;
        std::cout << "      Accuracy: " << acc << std::endl;
        std::cout << "      Precision: " << per_class_precision.at(class_label) << std::endl;
        std::cout << "      Recall: " << per_class_recall.at(class_label) << std::endl;
        std::cout << "      F1-Score: " << per_class_f1.at(class_label) << std::endl;
    }
}

void ClassificationMetrics::print_confusion_matrix() const {
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "      Predicted" << std::endl;
    std::cout << "Actual ";
    
    // Find max class label
    int max_class = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (confusion_matrix[i][j] > 0) {
                max_class = std::max(max_class, std::max(i, j));
            }
        }
    }
    
    // Print header
    for (int i = 0; i <= max_class; i++) {
        std::cout << std::setw(6) << i;
    }
    std::cout << std::endl;
    
    // Print matrix
    for (int i = 0; i <= max_class; i++) {
        std::cout << std::setw(6) << i;
        for (int j = 0; j <= max_class; j++) {
            std::cout << std::setw(6) << static_cast<int>(confusion_matrix[i][j]);
        }
        std::cout << std::endl;
    }
}

// TestResults implementation
void TestResults::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file << "total_samples=" << total_samples << std::endl;
    file << "correct_predictions=" << correct_predictions << std::endl;
    file << "testing_time_seconds=" << testing_time_seconds << std::endl;
    
    // Save predictions
    file << "predictions=";
    for (size_t i = 0; i < predictions.size(); i++) {
        file << predictions[i];
        if (i < predictions.size() - 1) file << ",";
    }
    file << std::endl;
    
    // Save confidence scores
    file << "confidence_scores=";
    for (size_t i = 0; i < confidence_scores.size(); i++) {
        file << confidence_scores[i];
        if (i < confidence_scores.size() - 1) file << ",";
    }
    file << std::endl;
    
    // Save metrics
    metrics.save(filename + ".metrics");
}

void TestResults::load(const std::string& filename) {
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
            
            if (key == "total_samples") total_samples = std::stoi(value);
            else if (key == "correct_predictions") correct_predictions = std::stoi(value);
            else if (key == "testing_time_seconds") testing_time_seconds = std::stod(value);
            else if (key == "predictions") {
                predictions.clear();
                std::istringstream iss(value);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    predictions.push_back(std::stoi(token));
                }
            }
            else if (key == "confidence_scores") {
                confidence_scores.clear();
                std::istringstream iss(value);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    confidence_scores.push_back(std::stod(token));
                }
            }
        }
    }
    
    // Load metrics
    metrics.load(filename + ".metrics");
}

void TestResults::print_summary() const {
    std::cout << "Test Results:" << std::endl;
    std::cout << "  Total samples: " << total_samples << std::endl;
    std::cout << "  Correct predictions: " << correct_predictions << std::endl;
    std::cout << "  Testing time: " << testing_time_seconds << " seconds" << std::endl;
    
    metrics.print_summary();
}

// LVQTester implementation
LVQTester::LVQTester(std::shared_ptr<LVQNetwork> network) : network_(network) {}

std::vector<double> LVQTester::calculate_confidence_scores(const std::vector<std::vector<double>>& features) {
    std::vector<double> scores;
    scores.reserve(features.size());
    
    for (const auto& feature : features) {
        auto prediction = network_->predict_with_confidence(feature);
        scores.push_back(prediction.second);
    }
    
    return scores;
}

void LVQTester::validate_test_data(const std::vector<DataPoint>& data) const {
    if (data.empty()) {
        throw std::invalid_argument("Test data is empty");
    }
    
    size_t expected_dim = data[0].features.size();
    for (const auto& data_point : data) {
        if (data_point.features.size() != expected_dim) {
            throw std::invalid_argument("Inconsistent feature dimensions in test data");
        }
    }
}

TestResults LVQTester::test(const std::vector<DataPoint>& test_data) {
    if (!network_) {
        throw std::runtime_error("No network assigned to tester");
    }
    
    if (!network_->is_trained()) {
        throw std::runtime_error("Network is not trained");
    }
    
    validate_test_data(test_data);
    
    Timer timer;
    timer.start();
    
    TestResults results;
    results.total_samples = test_data.size();
    results.predictions.reserve(test_data.size());
    
    int correct = 0;
    for (const auto& data_point : test_data) {
        int prediction = network_->predict(data_point.features);
        results.predictions.push_back(prediction);
        
        if (prediction == data_point.class_label) {
            correct++;
        }
    }
    
    results.correct_predictions = correct;
    
    timer.stop();
    results.testing_time_seconds = timer.elapsed_seconds();
    
    // Calculate metrics
    std::vector<int> true_labels;
    true_labels.reserve(test_data.size());
    for (const auto& data_point : test_data) {
        true_labels.push_back(data_point.class_label);
    }
    
    results.metrics.calculate_metrics(results.predictions, true_labels);
    
    return results;
}

TestResults LVQTester::test_with_confidence(const std::vector<DataPoint>& test_data) {
    if (!network_) {
        throw std::runtime_error("No network assigned to tester");
    }
    
    if (!network_->is_trained()) {
        throw std::runtime_error("Network is not trained");
    }
    
    validate_test_data(test_data);
    
    Timer timer;
    timer.start();
    
    TestResults results;
    results.total_samples = test_data.size();
    results.predictions.reserve(test_data.size());
    results.confidence_scores.reserve(test_data.size());
    
    int correct = 0;
    for (const auto& data_point : test_data) {
        auto prediction = network_->predict_with_confidence(data_point.features);
        results.predictions.push_back(prediction.first);
        results.confidence_scores.push_back(prediction.second);
        
        if (prediction.first == data_point.class_label) {
            correct++;
        }
    }
    
    results.correct_predictions = correct;
    
    timer.stop();
    results.testing_time_seconds = timer.elapsed_seconds();
    
    // Calculate metrics
    std::vector<int> true_labels;
    true_labels.reserve(test_data.size());
    for (const auto& data_point : test_data) {
        true_labels.push_back(data_point.class_label);
    }
    
    results.metrics.calculate_metrics(results.predictions, true_labels);
    
    return results;
}

TestResults LVQTester::test_batch(const std::vector<std::vector<double>>& features,
                                 const std::vector<int>& labels) {
    if (!network_) {
        throw std::runtime_error("No network assigned to tester");
    }
    
    if (!network_->is_trained()) {
        throw std::runtime_error("Network is not trained");
    }
    
    if (features.size() != labels.size()) {
        throw std::invalid_argument("Features and labels must have same size");
    }
    
    Timer timer;
    timer.start();
    
    TestResults results;
    results.total_samples = features.size();
    results.predictions = network_->predict_batch(features);
    results.confidence_scores = calculate_confidence_scores(features);
    
    int correct = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if (results.predictions[i] == labels[i]) {
            correct++;
        }
    }
    
    results.correct_predictions = correct;
    
    timer.stop();
    results.testing_time_seconds = timer.elapsed_seconds();
    
    // Calculate metrics
    results.metrics.calculate_metrics(results.predictions, labels);
    
    return results;
}

std::vector<TestResults> LVQTester::cross_validate_test(const std::vector<DataPoint>& data,
                                                       int num_folds) {
    if (!network_) {
        throw std::runtime_error("No network assigned to tester");
    }
    
    if (num_folds < 2 || num_folds > static_cast<int>(data.size())) {
        throw std::invalid_argument("Invalid number of folds");
    }
    
    std::vector<TestResults> results;
    std::vector<DataPoint> shuffled_data = data;
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), std::mt19937(42));
    
    int fold_size = data.size() / num_folds;
    
    for (int fold = 0; fold < num_folds; fold++) {
        // Split data into training and test
        std::vector<DataPoint> test_data;
        std::vector<DataPoint> train_data;
        
        int start_idx = fold * fold_size;
        int end_idx = (fold == num_folds - 1) ? data.size() : (fold + 1) * fold_size;
        
        for (size_t i = 0; i < data.size(); i++) {
            if (i >= start_idx && i < end_idx) {
                test_data.push_back(shuffled_data[i]);
            } else {
                train_data.push_back(shuffled_data[i]);
            }
        }
        
        // Create a new network for this fold
        auto fold_network = std::make_shared<LVQNetwork>(network_->get_config());
        fold_network->train(train_data);
        
        // Test on this fold
        LVQTester fold_tester(fold_network);
        auto fold_results = fold_tester.test(test_data);
        results.push_back(fold_results);
    }
    
    return results;
}

void LVQTester::analyze_performance(const std::vector<DataPoint>& test_data) {
    if (!network_) {
        throw std::runtime_error("No network assigned to tester");
    }
    
    auto results = test_with_confidence(test_data);
    
    std::cout << "Performance Analysis:" << std::endl;
    std::cout << "  Average confidence: " << std::accumulate(results.confidence_scores.begin(), 
                                                            results.confidence_scores.end(), 0.0) / 
                                            results.confidence_scores.size() << std::endl;
    
    // Find samples with low confidence
    std::vector<size_t> low_confidence_indices;
    for (size_t i = 0; i < results.confidence_scores.size(); i++) {
        if (results.confidence_scores[i] < 0.5) {
            low_confidence_indices.push_back(i);
        }
    }
    
    std::cout << "  Low confidence predictions (< 0.5): " << low_confidence_indices.size() << std::endl;
}

void LVQTester::generate_error_analysis(const std::vector<DataPoint>& test_data,
                                       const std::vector<int>& predictions) {
    if (test_data.size() != predictions.size()) {
        throw std::invalid_argument("Test data and predictions must have same size");
    }
    
    std::cout << "Error Analysis:" << std::endl;
    
    // Find misclassified samples
    std::vector<size_t> error_indices;
    for (size_t i = 0; i < test_data.size(); i++) {
        if (predictions[i] != test_data[i].class_label) {
            error_indices.push_back(i);
        }
    }
    
    std::cout << "  Total errors: " << error_indices.size() << std::endl;
    std::cout << "  Error rate: " << static_cast<double>(error_indices.size()) / test_data.size() << std::endl;
    
    // Show some example errors
    std::cout << "  Example errors:" << std::endl;
    for (size_t i = 0; i < std::min(error_indices.size(), size_t(5)); i++) {
        size_t idx = error_indices[i];
        std::cout << "    Sample " << idx << ": True=" << test_data[idx].class_label 
                  << ", Predicted=" << predictions[idx] << std::endl;
    }
}

std::map<std::string, TestResults> LVQTester::compare_models(
    const std::map<std::string, std::shared_ptr<LVQNetwork>>& models,
    const std::vector<DataPoint>& test_data) {
    
    std::map<std::string, TestResults> results;
    
    for (const auto& [name, model] : models) {
        LVQTester tester(model);
        results[name] = tester.test(test_data);
    }
    
    return results;
}

bool LVQTester::validate_network() const {
    return network_ != nullptr && network_->is_trained();
}

std::vector<std::pair<int, double>> LVQTester::get_top_k_predictions(
    const std::vector<double>& features, int k) const {
    
    if (!network_) {
        throw std::runtime_error("No network assigned to tester");
    }
    
    if (!network_->is_trained()) {
        throw std::runtime_error("Network is not trained");
    }
    
    // Get all codebook vectors and their distances
    std::vector<std::pair<int, double>> distances;
    const auto& codebook_vectors = network_->get_codebook_vectors();
    
    for (const auto& codebook : codebook_vectors) {
        double distance = math_utils::euclidean_distance(features, codebook.weights);
        distances.emplace_back(codebook.class_label, distance);
    }
    
    // Sort by distance (ascending)
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Take top k and convert distances to confidence scores
    std::vector<std::pair<int, double>> top_k;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); i++) {
        double confidence = 1.0 / (1.0 + distances[i].second);
        top_k.emplace_back(distances[i].first, confidence);
    }
    
    return top_k;
}

} // namespace lvq 