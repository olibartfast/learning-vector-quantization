#pragma once

#include "lvq_network.h"
#include "utils.h"
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <iostream>

namespace lvq {

/**
 * @brief Dataset information and metadata
 */
struct DatasetInfo {
    std::string name;
    std::string description;
    int num_samples;
    int num_features;
    int num_classes;
    std::vector<std::string> class_names;
    std::vector<int> class_distribution;
    
    void print_summary() const;
};

/**
 * @brief Data preprocessing options
 */
struct PreprocessingOptions {
    bool normalize_features = true;
    bool standardize_features = false;
    bool shuffle_data = true;
    double train_split_ratio = 0.8;
    double validation_split_ratio = 0.1;
    int random_seed = 42;
    bool remove_outliers = false;
    double outlier_threshold = 3.0;
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};

/**
 * @brief Data loader for different dataset formats
 */
class DataLoader {
private:
    PreprocessingOptions options_;
    std::mt19937 rng_;
    
    // Helper methods
    std::vector<double> normalize_features(const std::vector<double>& features,
                                          const std::vector<double>& mean,
                                          const std::vector<double>& std_dev);
    std::vector<double> standardize_features(const std::vector<double>& features,
                                            const std::vector<double>& mean,
                                            const std::vector<double>& std_dev);
    std::vector<DataPoint> remove_outliers(const std::vector<DataPoint>& data);
    std::vector<DataPoint> shuffle_data(const std::vector<DataPoint>& data);
    
public:
    // Constructors
    DataLoader() = default;
    explicit DataLoader(const PreprocessingOptions& options);
    
    // Destructor
    ~DataLoader() = default;
    
    // Dataset loading methods
    std::vector<DataPoint> load_csv(const std::string& filename, 
                                   int label_column = -1,
                                   bool has_header = true);
    std::vector<DataPoint> load_mnist(const std::string& images_file,
                                     const std::string& labels_file);
    std::vector<DataPoint> load_iris(const std::string& filename = "");
    std::vector<DataPoint> load_custom_format(const std::string& filename,
                                             const std::string& format);
    
    // Data splitting
    struct DataSplit {
        std::vector<DataPoint> train;
        std::vector<DataPoint> validation;
        std::vector<DataPoint> test;
    };
    
    DataSplit split_data(const std::vector<DataPoint>& data);
    DataSplit split_data(const std::vector<DataPoint>& data,
                        double train_ratio,
                        double validation_ratio);
    
    // Data preprocessing
    std::vector<DataPoint> preprocess_data(const std::vector<DataPoint>& data);
    std::vector<DataPoint> normalize_data(const std::vector<DataPoint>& data);
    std::vector<DataPoint> standardize_data(const std::vector<DataPoint>& data);
    
    // Dataset information
    DatasetInfo get_dataset_info(const std::vector<DataPoint>& data) const;
    void print_data_statistics(const std::vector<DataPoint>& data) const;
    
    // Data export
    void save_data(const std::vector<DataPoint>& data, 
                   const std::string& filename,
                   const std::string& format = "csv");
    
    // Getters and setters
    void set_options(const PreprocessingOptions& options) { options_ = options; }
    const PreprocessingOptions& get_options() const { return options_; }
    
    // Utility methods
    bool validate_data(const std::vector<DataPoint>& data) const;
    std::vector<DataPoint> sample_data(const std::vector<DataPoint>& data, 
                                       int num_samples);
};

/**
 * @brief MNIST specific data loader
 */
class MNISTLoader : public DataLoader {
private:
    std::string data_dir_;
    
    // MNIST file reading helpers
    std::vector<std::vector<double>> read_mnist_images(const std::string& filename);
    std::vector<int> read_mnist_labels(const std::string& filename);
    int reverse_int(int i);
    
public:
    // Constructors
    MNISTLoader() = default;
    explicit MNISTLoader(const std::string& data_dir);
    MNISTLoader(const std::string& data_dir, const PreprocessingOptions& options);
    
    // MNIST specific loading
    std::vector<DataPoint> load_training_data();
    std::vector<DataPoint> load_test_data();
    std::vector<DataPoint> load_full_dataset();
    
    // MNIST preprocessing
    std::vector<DataPoint> preprocess_mnist(const std::vector<DataPoint>& data);
    std::vector<double> flatten_image(const std::vector<std::vector<double>>& image);
    
    // Getters and setters
    void set_data_dir(const std::string& data_dir) { data_dir_ = data_dir; }
    const std::string& get_data_dir() const { return data_dir_; }
};

/**
 * @brief Iris dataset specific data loader
 */
class IrisLoader : public DataLoader {
private:
    std::string data_file_;
    std::map<std::string, int> class_mapping_;
    
    // Iris specific helpers
    void initialize_class_mapping();
    std::vector<DataPoint> load_iris_from_file(const std::string& filename);
    std::vector<DataPoint> load_iris_builtin();
    
public:
    // Constructors
    IrisLoader() = default;
    explicit IrisLoader(const std::string& data_file);
    IrisLoader(const std::string& data_file, const PreprocessingOptions& options);
    
    // Iris specific loading
    std::vector<DataPoint> load_full_dataset();
    std::vector<DataPoint> load_training_data();
    std::vector<DataPoint> load_test_data();
    
    // Iris preprocessing
    std::vector<DataPoint> preprocess_iris(const std::vector<DataPoint>& data);
    
    // Getters and setters
    void set_data_file(const std::string& data_file) { data_file_ = data_file; }
    const std::string& get_data_file() const { return data_file_; }
    const std::map<std::string, int>& get_class_mapping() const { return class_mapping_; }
    
    // Utility methods
    std::string get_class_name(int class_id) const;
    int get_class_id(const std::string& class_name) const;
};

} // namespace lvq 