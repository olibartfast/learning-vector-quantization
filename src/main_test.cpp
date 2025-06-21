#include "lvq_network.h"
#include "lvq_trainer.h"
#include "lvq_tester.h"
#include "data_loader.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <memory>

using namespace lvq;

void print_usage() {
    std::cout << "LVQ Network Testing Tool\n";
    std::cout << "Usage: lvq_test [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --model <file>       Trained model file\n";
    std::cout << "  --dataset <path>     Test dataset path\n";
    std::cout << "  --output <dir>       Output directory for results\n";
    std::cout << "  --detailed           Generate detailed analysis\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string model_file = "models/lvq_model.bin";
    std::string dataset_path = "data/mnist";
    std::string output_dir = "results";
    bool detailed_analysis = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--detailed") {
            detailed_analysis = true;
        }
    }
    
    try {
        // Initialize logging
        log_utils::set_log_level(log_utils::LogLevel::INFO);
        log_utils::log_info("Starting LVQ Network Testing");
        
        // Check if model file exists
        if (!file_utils::file_exists(model_file)) {
            log_utils::log_error("Model file not found: " + model_file);
            return 1;
        }
        
        // Create output directory
        if (!file_utils::create_directory(output_dir)) {
            log_utils::log_warning("Could not create output directory: " + output_dir);
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
        
        // Load test dataset
        log_utils::log_info("Loading test dataset from: " + dataset_path);
        MNISTLoader data_loader(dataset_path);
        
        PreprocessingOptions preprocess_opts;
        preprocess_opts.normalize_features = true;
        preprocess_opts.shuffle_data = false; // Keep test data order
        data_loader.set_options(preprocess_opts);
        
        auto test_data = data_loader.load_test_data();
        test_data = data_loader.preprocess_data(test_data);
        
        log_utils::log_info("Loaded " + std::to_string(test_data.size()) + " test samples");
        
        // Create tester
        LVQTester tester(network);
        
        // Perform testing
        log_utils::log_info("Starting testing...");
        Timer timer;
        timer.start();
        
        TestResults test_results;
        if (detailed_analysis) {
            test_results = tester.test_with_confidence(test_data);
        } else {
            test_results = tester.test(test_data);
        }
        
        timer.stop();
        log_utils::log_info("Testing completed in " + std::to_string(timer.elapsed_seconds()) + " seconds");
        
        // Print results
        log_utils::log_info("Test Results:");
        log_utils::log_info("  Total samples: " + std::to_string(test_results.total_samples));
        log_utils::log_info("  Correct predictions: " + std::to_string(test_results.correct_predictions));
        log_utils::log_info("  Accuracy: " + std::to_string(test_results.metrics.accuracy));
        log_utils::log_info("  Precision: " + std::to_string(test_results.metrics.precision));
        log_utils::log_info("  Recall: " + std::to_string(test_results.metrics.recall));
        log_utils::log_info("  F1-Score: " + std::to_string(test_results.metrics.f1_score));
        
        // Print per-class metrics
        log_utils::log_info("Per-class accuracy:");
        for (const auto& [class_label, accuracy] : test_results.metrics.per_class_accuracy) {
            log_utils::log_info("  Class " + std::to_string(class_label) + ": " + std::to_string(accuracy));
        }
        
        // Print confusion matrix
        log_utils::log_info("Confusion Matrix:");
        test_results.metrics.print_confusion_matrix();
        
        // Save results
        std::string results_file = output_dir + "/test_results.txt";
        test_results.save(results_file);
        log_utils::log_info("Results saved to: " + results_file);
        
        // Generate detailed analysis if requested
        if (detailed_analysis) {
            log_utils::log_info("Generating detailed analysis...");
            
            // Error analysis
            tester.generate_error_analysis(test_data, test_results.predictions);
            
            // Performance analysis
            tester.analyze_performance(test_data);
            
            // Save confidence scores
            if (!test_results.confidence_scores.empty()) {
                std::string confidence_file = output_dir + "/confidence_scores.txt";
                std::ofstream conf_file(confidence_file);
                if (conf_file.is_open()) {
                    conf_file << "Sample,Prediction,True_Label,Confidence\n";
                    for (size_t i = 0; i < test_data.size(); i++) {
                        conf_file << i << "," 
                                 << test_results.predictions[i] << ","
                                 << test_data[i].class_label << ","
                                 << test_results.confidence_scores[i] << "\n";
                    }
                    conf_file.close();
                    log_utils::log_info("Confidence scores saved to: " + confidence_file);
                }
            }
        }
        
        // Save predictions
        std::string predictions_file = output_dir + "/predictions.txt";
        std::ofstream pred_file(predictions_file);
        if (pred_file.is_open()) {
            pred_file << "Sample,Prediction,True_Label\n";
            for (size_t i = 0; i < test_data.size(); i++) {
                pred_file << i << "," 
                          << test_results.predictions[i] << ","
                          << test_data[i].class_label << "\n";
            }
            pred_file.close();
            log_utils::log_info("Predictions saved to: " + predictions_file);
        }
        
        log_utils::log_info("Testing completed successfully!");
        
    } catch (const std::exception& e) {
        log_utils::log_error("Error during testing: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
} 