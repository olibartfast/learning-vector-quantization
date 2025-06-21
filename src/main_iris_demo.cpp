#include "lvq_network.h"
#include "lvq_trainer.h"
#include "lvq_tester.h"
#include "data_loader.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <iomanip>

using namespace lvq;

void print_usage() {
    std::cout << "LVQ Network Iris Dataset Demo\n";
    std::cout << "Usage: lvq_iris_demo [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --train              Train a new model on Iris dataset\n";
    std::cout << "  --model <file>       Load existing model file\n";
    std::cout << "  --output <dir>       Output directory for results\n";
    std::cout << "  --epochs <num>       Number of training epochs\n";
    std::cout << "  --lr <rate>          Learning rate\n";
    std::cout << "  --codebook <num>     Number of codebook vectors per class\n";
    std::cout << "  --interactive        Enter interactive prediction mode\n";
    std::cout << "  --help               Show this help message\n";
}

void print_iris_info() {
    std::cout << "\n=== Iris Dataset Information ===\n";
    std::cout << "The Iris dataset contains 150 samples of iris flowers with 4 features:\n";
    std::cout << "1. Sepal length (cm)\n";
    std::cout << "2. Sepal width (cm)\n";
    std::cout << "3. Petal length (cm)\n";
    std::cout << "4. Petal width (cm)\n\n";
    std::cout << "Three classes:\n";
    std::cout << "- Setosa (class 0)\n";
    std::cout << "- Versicolor (class 1)\n";
    std::cout << "- Virginica (class 2)\n\n";
}

void print_feature_ranges() {
    std::cout << "Typical feature ranges:\n";
    std::cout << "Sepal length: 4.3 - 7.9 cm\n";
    std::cout << "Sepal width:  2.0 - 4.4 cm\n";
    std::cout << "Petal length: 1.0 - 6.9 cm\n";
    std::cout << "Petal width:  0.1 - 2.5 cm\n\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool train_model = false;
    std::string model_file = "models/iris_model.bin";
    std::string output_dir = "results/iris";
    int epochs = 100;
    double learning_rate = 0.01;
    int codebook_vectors = 5;
    bool interactive_mode = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--train") {
            train_model = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stod(argv[++i]);
        } else if (arg == "--codebook" && i + 1 < argc) {
            codebook_vectors = std::stoi(argv[++i]);
        } else if (arg == "--interactive") {
            interactive_mode = true;
        }
    }
    
    try {
        // Initialize logging
        log_utils::set_log_level(log_utils::LogLevel::INFO);
        log_utils::log_info("Starting LVQ Network Iris Dataset Demo");
        
        // Create output directory
        if (!file_utils::create_directory(output_dir)) {
            log_utils::log_warning("Could not create output directory: " + output_dir);
        }
        
        // Create models directory if needed
        std::string models_dir = "models";
        if (!file_utils::create_directory(models_dir)) {
            log_utils::log_warning("Could not create models directory");
        }
        
        std::shared_ptr<LVQNetwork> network;
        
        if (train_model) {
            // Train a new model
            log_utils::log_info("Training new model on Iris dataset...");
            
            // Configure network
            LVQConfig config;
            config.learning_rate = learning_rate;
            config.num_codebook_vectors = codebook_vectors;
            config.max_iterations = epochs;
            config.random_seed = 42;
            
            log_utils::log_info("Configuration:");
            log_utils::log_info("  Learning rate: " + std::to_string(config.learning_rate));
            log_utils::log_info("  Codebook vectors per class: " + std::to_string(config.num_codebook_vectors));
            log_utils::log_info("  Max iterations: " + std::to_string(config.max_iterations));
            
            // Load Iris dataset
            log_utils::log_info("Loading Iris dataset...");
            IrisLoader data_loader;
            
            PreprocessingOptions preprocess_opts;
            preprocess_opts.normalize_features = true;
            preprocess_opts.shuffle_data = true;
            preprocess_opts.train_split_ratio = 0.8;
            preprocess_opts.validation_split_ratio = 0.1;
            data_loader.set_options(preprocess_opts);
            
            auto full_dataset = data_loader.load_full_dataset();
            log_utils::log_info("Loaded " + std::to_string(full_dataset.size()) + " Iris samples");
            
            // Print dataset info
            auto dataset_info = data_loader.get_dataset_info(full_dataset);
            dataset_info.print_summary();
            
            // Split data
            auto data_split = data_loader.split_data(full_dataset);
            log_utils::log_info("Training samples: " + std::to_string(data_split.train.size()));
            log_utils::log_info("Validation samples: " + std::to_string(data_split.validation.size()));
            log_utils::log_info("Test samples: " + std::to_string(data_split.test.size()));
            
            // Preprocess data
            auto train_data = data_loader.preprocess_data(data_split.train);
            auto validation_data = data_loader.preprocess_data(data_split.validation);
            auto test_data = data_loader.preprocess_data(data_split.test);
            
            // Create and train network
            network = std::make_shared<LVQNetwork>(config);
            LVQTrainer trainer(network, config);
            
            // Set up training callbacks
            TrainingCallbacks callbacks;
            callbacks.on_iteration = [](int iteration, double error, double lr) {
                if (iteration % 10 == 0) {
                    log_utils::log_info("Iteration " + std::to_string(iteration) + 
                                       ", Error: " + std::to_string(error) + 
                                       ", LR: " + std::to_string(lr));
                }
            };
            
            callbacks.on_complete = [](const TrainingStats& stats) {
                log_utils::log_info("Training completed!");
                log_utils::log_info("Final error: " + std::to_string(stats.final_error));
                log_utils::log_info("Training time: " + std::to_string(stats.training_time_seconds) + " seconds");
            };
            
            trainer.set_callbacks(callbacks);
            
            // Train the network - use regular training instead of validation training
            log_utils::log_info("=== TRAINING START ===");
            log_utils::log_info("Starting training...");
            Timer timer;
            timer.start();
            
            // Use regular training since we have too few samples for validation
            auto training_stats = trainer.train(train_data);
            
            timer.stop();
            log_utils::log_info("Training completed in " + std::to_string(timer.elapsed_seconds()) + " seconds");
            log_utils::log_info("=== TRAINING END ===");
            
            // Save training statistics
            training_stats.save(output_dir + "/iris_training_stats.txt");
            
            // Test the network
            log_utils::log_info("=== TESTING START ===");
            log_utils::log_info("Testing network...");
            LVQTester tester(network);
            auto test_results = tester.test_with_confidence(test_data);
            log_utils::log_info("=== TESTING END ===");
            
            log_utils::log_info("Test Results:");
            log_utils::log_info("  Accuracy: " + std::to_string(test_results.metrics.accuracy));
            log_utils::log_info("  Precision: " + std::to_string(test_results.metrics.precision));
            log_utils::log_info("  Recall: " + std::to_string(test_results.metrics.recall));
            log_utils::log_info("  F1-Score: " + std::to_string(test_results.metrics.f1_score));
            
            // Print per-class results
            log_utils::log_info("Per-class accuracy:");
            for (const auto& [class_label, accuracy] : test_results.metrics.per_class_accuracy) {
                std::string class_name = data_loader.get_class_name(class_label);
                log_utils::log_info("  " + class_name + ": " + std::to_string(accuracy));
            }
            
            // Print confusion matrix
            log_utils::log_info("Confusion Matrix:");
            test_results.metrics.print_confusion_matrix();
            
            // Save test results
            test_results.save(output_dir + "/iris_test_results.txt");
            
            // Save the trained model
            network->save_model(model_file);
            log_utils::log_info("Model saved to: " + model_file);
            
            // Save configuration
            config.save(output_dir + "/iris_config.txt");
            
            // Print summary
            network->print_summary();
            
        } else {
            // Load existing model
            if (!file_utils::file_exists(model_file)) {
                log_utils::log_error("Model file not found: " + model_file);
                log_utils::log_info("Use --train to train a new model first.");
                return 1;
            }
            
            log_utils::log_info("Loading model from: " + model_file);
            network = std::make_shared<LVQNetwork>();
            network->load_model(model_file);
            
            if (!network->is_trained()) {
                log_utils::log_error("Loaded model is not trained!");
                return 1;
            }
            
            log_utils::log_info("Model loaded successfully");
            network->print_summary();
        }
        
        // Interactive mode
        if (interactive_mode) {
            print_iris_info();
            print_feature_ranges();
            
            log_utils::log_info("Entering interactive mode. Enter 'quit' to exit.");
            log_utils::log_info("Enter 4 feature values (sepal length, sepal width, petal length, petal width):");
            
            IrisLoader data_loader;
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
                
                if (features.size() != 4) {
                    log_utils::log_error("Expected 4 features, got " + std::to_string(features.size()));
                    log_utils::log_info("Please enter: sepal_length sepal_width petal_length petal_width");
                    continue;
                }
                
                // Validate feature ranges
                if (features[0] < 4.0 || features[0] > 8.0 ||
                    features[1] < 2.0 || features[1] > 5.0 ||
                    features[2] < 1.0 || features[2] > 7.0 ||
                    features[3] < 0.1 || features[3] > 3.0) {
                    log_utils::log_warning("Features outside typical ranges. Prediction may be unreliable.");
                }
                
                // Make prediction
                auto prediction = network->predict_with_confidence(features);
                std::string class_name = data_loader.get_class_name(prediction.first);
                
                std::cout << "\nPrediction: " << class_name 
                          << " (confidence: " << std::fixed << std::setprecision(3) << prediction.second << ")" << std::endl;
                
                // Show top-3 predictions
                LVQTester tester(network);
                auto top_predictions = tester.get_top_k_predictions(features, 3);
                std::cout << "Top-3 predictions: ";
                for (const auto& [pred, conf] : top_predictions) {
                    std::string pred_name = data_loader.get_class_name(pred);
                    std::cout << pred_name << " (" << std::fixed << std::setprecision(3) << conf << ") ";
                }
                std::cout << std::endl;
                
                // Show feature values
                std::cout << "Input features: ";
                std::cout << "Sepal length=" << std::fixed << std::setprecision(1) << features[0] << "cm, ";
                std::cout << "Sepal width=" << features[1] << "cm, ";
                std::cout << "Petal length=" << features[2] << "cm, ";
                std::cout << "Petal width=" << features[3] << "cm" << std::endl;
                std::cout << std::endl;
            }
        }
        
        log_utils::log_info("Iris demo completed successfully!");
        
    } catch (const std::exception& e) {
        log_utils::log_error("Error during Iris demo: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
} 