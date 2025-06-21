#include "lvq_network.h"
#include "lvq_trainer.h"
#include "lvq_tester.h"
#include "data_loader.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <memory>
#include <vector>

using namespace lvq;

void print_usage() {
    std::cout << "LVQ Network Demo Tool\n";
    std::cout << "Usage: lvq_demo [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --model <file>       Trained model file\n";
    std::cout << "  --input <file>       Input data file (CSV format)\n";
    std::cout << "  --output <file>      Output predictions file\n";
    std::cout << "  --confidence         Include confidence scores\n";
    std::cout << "  --top-k <num>        Show top-k predictions\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string model_file = "models/lvq_model.bin";
    std::string input_file = "";
    std::string output_file = "predictions.csv";
    bool include_confidence = false;
    int top_k = 1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--confidence") {
            include_confidence = true;
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        }
    }
    
    try {
        // Initialize logging
        log_utils::set_log_level(log_utils::LogLevel::INFO);
        log_utils::log_info("Starting LVQ Network Demo");
        
        // Check if model file exists
        if (!file_utils::file_exists(model_file)) {
            log_utils::log_error("Model file not found: " + model_file);
            return 1;
        }
        
        // Load the trained model
        log_utils::log_info("Loading model from: " + model_file);
        auto network = std::make_shared<LVQNetwork>();
        network->load_model(model_file);
        
        if (!network->is_trained()) {
            log_utils::log_error("Loaded model is not trained!");
            return 1;
        }
        
        log_utils::log_info("Model loaded successfully");
        network->print_summary();
        
        // Load input data
        std::vector<DataPoint> input_data;
        if (!input_file.empty()) {
            log_utils::log_info("Loading input data from: " + input_file);
            DataLoader data_loader;
            input_data = data_loader.load_csv(input_file, -1, true); // No label column, has header
            log_utils::log_info("Loaded " + std::to_string(input_data.size()) + " samples");
        } else {
            // Generate some sample data for demonstration
            log_utils::log_info("No input file provided, generating sample data...");
            size_t input_dim = network->get_input_dimension();
            int num_classes = network->get_unique_classes().size();
            
            for (int i = 0; i < 10; i++) {
                std::vector<double> features(input_dim);
                for (size_t j = 0; j < input_dim; j++) {
                    features[j] = random_utils::get_random_double(0.0, 1.0);
                }
                input_data.emplace_back(features, i % num_classes);
            }
            log_utils::log_info("Generated " + std::to_string(input_data.size()) + " sample data points");
        }
        
        // Preprocess input data
        PreprocessingOptions preprocess_opts;
        preprocess_opts.normalize_features = true;
        preprocess_opts.shuffle_data = false;
        
        DataLoader data_loader(preprocess_opts);
        auto processed_data = data_loader.preprocess_data(input_data);
        
        // Create tester for predictions
        LVQTester tester(network);
        
        // Make predictions
        log_utils::log_info("Making predictions...");
        Timer timer;
        timer.start();
        
        std::vector<int> predictions;
        std::vector<double> confidence_scores;
        
        if (include_confidence) {
            auto test_results = tester.test_with_confidence(processed_data);
            predictions = test_results.predictions;
            confidence_scores = test_results.confidence_scores;
        } else {
            auto test_results = tester.test(processed_data);
            predictions = test_results.predictions;
        }
        
        timer.stop();
        log_utils::log_info("Predictions completed in " + std::to_string(timer.elapsed_milliseconds()) + " ms");
        
        // Display results
        log_utils::log_info("Prediction Results:");
        for (size_t i = 0; i < processed_data.size(); i++) {
            std::cout << "Sample " << i << ": ";
            
            if (top_k > 1) {
                auto top_predictions = tester.get_top_k_predictions(processed_data[i].features, top_k);
                std::cout << "Top-" << top_k << " predictions: ";
                for (const auto& [pred, conf] : top_predictions) {
                    std::cout << "Class " << pred << " (" << std::fixed << std::setprecision(3) << conf << ") ";
                }
            } else {
                std::cout << "Predicted class: " << predictions[i];
                if (include_confidence && i < confidence_scores.size()) {
                    std::cout << " (confidence: " << std::fixed << std::setprecision(3) << confidence_scores[i] << ")";
                }
            }
            std::cout << std::endl;
        }
        
        // Save predictions to file
        log_utils::log_info("Saving predictions to: " + output_file);
        std::ofstream out_file(output_file);
        if (out_file.is_open()) {
            // Write header
            out_file << "Sample";
            if (top_k > 1) {
                for (int k = 1; k <= top_k; k++) {
                    out_file << ",Top" << k << "_Class,Top" << k << "_Confidence";
                }
            } else {
                out_file << ",Predicted_Class";
                if (include_confidence) {
                    out_file << ",Confidence";
                }
            }
            out_file << "\n";
            
            // Write predictions
            for (size_t i = 0; i < processed_data.size(); i++) {
                out_file << i;
                
                if (top_k > 1) {
                    auto top_predictions = tester.get_top_k_predictions(processed_data[i].features, top_k);
                    for (int k = 0; k < top_k; k++) {
                        if (k < top_predictions.size()) {
                            out_file << "," << top_predictions[k].first 
                                   << "," << std::fixed << std::setprecision(6) << top_predictions[k].second;
                        } else {
                            out_file << ",-1,0.0";
                        }
                    }
                } else {
                    out_file << "," << predictions[i];
                    if (include_confidence && i < confidence_scores.size()) {
                        out_file << "," << std::fixed << std::setprecision(6) << confidence_scores[i];
                    }
                }
                out_file << "\n";
            }
            out_file.close();
            log_utils::log_info("Predictions saved successfully");
        } else {
            log_utils::log_error("Could not open output file: " + output_file);
        }
        
        // Interactive mode for single predictions
        if (input_file.empty()) {
            log_utils::log_info("Entering interactive mode. Enter 'quit' to exit.");
            log_utils::log_info("Enter feature values separated by spaces (dimension: " + 
                               std::to_string(network->get_input_dimension()) + "):");
            
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "quit" || line == "exit") {
                    break;
                }
                
                // Parse input features
                std::vector<double> features;
                std::istringstream iss(line);
                double value;
                while (iss >> value) {
                    features.push_back(value);
                }
                
                if (features.size() != network->get_input_dimension()) {
                    log_utils::log_error("Expected " + std::to_string(network->get_input_dimension()) + 
                                        " features, got " + std::to_string(features.size()));
                    continue;
                }
                
                // Make prediction
                auto prediction = network->predict_with_confidence(features);
                std::cout << "Prediction: Class " << prediction.first 
                          << " (confidence: " << std::fixed << std::setprecision(3) << prediction.second << ")" << std::endl;
                
                if (top_k > 1) {
                    auto top_predictions = tester.get_top_k_predictions(features, top_k);
                    std::cout << "Top-" << top_k << " predictions: ";
                    for (const auto& [pred, conf] : top_predictions) {
                        std::cout << "Class " << pred << " (" << std::fixed << std::setprecision(3) << conf << ") ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        
        log_utils::log_info("Demo completed successfully!");
        
    } catch (const std::exception& e) {
        log_utils::log_error("Error during demo: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
} 