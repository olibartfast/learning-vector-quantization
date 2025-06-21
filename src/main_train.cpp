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
    std::cout << "LVQ Network Training Tool\n";
    std::cout << "Usage: lvq_train [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --dataset <path>     Dataset path (default: data/mnist)\n";
    std::cout << "  --config <file>      Configuration file\n";
    std::cout << "  --output <dir>       Output directory for models\n";
    std::cout << "  --epochs <num>       Number of training epochs\n";
    std::cout << "  --lr <rate>          Learning rate\n";
    std::cout << "  --codebook <num>     Number of codebook vectors per class\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string dataset_path = "data/mnist";
    std::string config_file = "";
    std::string output_dir = "models";
    int epochs = 100;
    double learning_rate = 0.01;
    int codebook_vectors = 10;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stod(argv[++i]);
        } else if (arg == "--codebook" && i + 1 < argc) {
            codebook_vectors = std::stoi(argv[++i]);
        }
    }
    
    try {
        // Initialize logging
        log_utils::set_log_level(log_utils::LogLevel::INFO);
        log_utils::log_info("Starting LVQ Network Training");
        
        // Create output directory
        if (!file_utils::create_directory(output_dir)) {
            log_utils::log_warning("Could not create output directory: " + output_dir);
        }
        
        // Load configuration if provided
        LVQConfig config;
        if (!config_file.empty()) {
            auto config_map = config_utils::load_config(config_file);
            config.learning_rate = config_utils::get_config_double(config_map, "learning_rate", learning_rate);
            config.num_codebook_vectors = config_utils::get_config_int(config_map, "num_codebook_vectors", codebook_vectors);
            config.max_iterations = config_utils::get_config_int(config_map, "max_iterations", epochs);
        } else {
            config.learning_rate = learning_rate;
            config.num_codebook_vectors = codebook_vectors;
            config.max_iterations = epochs;
        }
        
        log_utils::log_info("Configuration loaded:");
        log_utils::log_info("  Learning rate: " + std::to_string(config.learning_rate));
        log_utils::log_info("  Codebook vectors per class: " + std::to_string(config.num_codebook_vectors));
        log_utils::log_info("  Max iterations: " + std::to_string(config.max_iterations));
        
        // Load dataset
        log_utils::log_info("Loading dataset from: " + dataset_path);
        MNISTLoader data_loader(dataset_path);
        
        PreprocessingOptions preprocess_opts;
        preprocess_opts.normalize_features = true;
        preprocess_opts.shuffle_data = true;
        preprocess_opts.train_split_ratio = 0.8;
        preprocess_opts.validation_split_ratio = 0.1;
        data_loader.set_options(preprocess_opts);
        
        auto full_dataset = data_loader.load_full_dataset();
        log_utils::log_info("Loaded " + std::to_string(full_dataset.size()) + " samples");
        
        // Split data
        auto data_split = data_loader.split_data(full_dataset);
        log_utils::log_info("Training samples: " + std::to_string(data_split.train.size()));
        log_utils::log_info("Validation samples: " + std::to_string(data_split.validation.size()));
        log_utils::log_info("Test samples: " + std::to_string(data_split.test.size()));
        
        // Preprocess data
        auto train_data = data_loader.preprocess_data(data_split.train);
        auto validation_data = data_loader.preprocess_data(data_split.validation);
        auto test_data = data_loader.preprocess_data(data_split.test);
        
        // Create network
        auto network = std::make_shared<LVQNetwork>(config);
        log_utils::log_info("LVQ Network created");
        
        // Create trainer
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
        
        // Train the network
        log_utils::log_info("Starting training...");
        Timer timer;
        timer.start();
        
        auto training_stats = trainer.train_with_validation(train_data, validation_data);
        
        timer.stop();
        log_utils::log_info("Training completed in " + std::to_string(timer.elapsed_seconds()) + " seconds");
        
        // Save training statistics
        training_stats.save(output_dir + "/training_stats.txt");
        
        // Test the network
        log_utils::log_info("Testing network...");
        LVQTester tester(network);
        auto test_results = tester.test(test_data);
        
        log_utils::log_info("Test Results:");
        log_utils::log_info("  Accuracy: " + std::to_string(test_results.metrics.accuracy));
        log_utils::log_info("  Precision: " + std::to_string(test_results.metrics.precision));
        log_utils::log_info("  Recall: " + std::to_string(test_results.metrics.recall));
        log_utils::log_info("  F1-Score: " + std::to_string(test_results.metrics.f1_score));
        
        // Save test results
        test_results.save(output_dir + "/test_results.txt");
        
        // Save the trained model
        std::string model_file = output_dir + "/lvq_model.bin";
        network->save_model(model_file);
        log_utils::log_info("Model saved to: " + model_file);
        
        // Save configuration
        config.save(output_dir + "/config.txt");
        
        // Print summary
        network->print_summary();
        
        log_utils::log_info("Training completed successfully!");
        
    } catch (const std::exception& e) {
        log_utils::log_error("Error during training: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
} 