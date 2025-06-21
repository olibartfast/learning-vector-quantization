#include "data_loader.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <set>
#include <map>

namespace lvq {

// DatasetInfo implementation
void DatasetInfo::print_summary() const {
    std::cout << "Dataset Information:" << std::endl;
    std::cout << "  Name: " << name << std::endl;
    std::cout << "  Description: " << description << std::endl;
    std::cout << "  Number of samples: " << num_samples << std::endl;
    std::cout << "  Number of features: " << num_features << std::endl;
    std::cout << "  Number of classes: " << num_classes << std::endl;
    
    if (!class_names.empty()) {
        std::cout << "  Class names: ";
        for (size_t i = 0; i < class_names.size(); i++) {
            std::cout << class_names[i];
            if (i < class_names.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    if (!class_distribution.empty()) {
        std::cout << "  Class distribution: ";
        for (size_t i = 0; i < class_distribution.size(); i++) {
            std::cout << class_distribution[i];
            if (i < class_distribution.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}

// PreprocessingOptions implementation
void PreprocessingOptions::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file << "normalize_features=" << (normalize_features ? "true" : "false") << std::endl;
    file << "standardize_features=" << (standardize_features ? "true" : "false") << std::endl;
    file << "shuffle_data=" << (shuffle_data ? "true" : "false") << std::endl;
    file << "train_split_ratio=" << train_split_ratio << std::endl;
    file << "validation_split_ratio=" << validation_split_ratio << std::endl;
    file << "random_seed=" << random_seed << std::endl;
    file << "remove_outliers=" << (remove_outliers ? "true" : "false") << std::endl;
    file << "outlier_threshold=" << outlier_threshold << std::endl;
}

void PreprocessingOptions::load(const std::string& filename) {
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
            
            if (key == "normalize_features") normalize_features = (value == "true");
            else if (key == "standardize_features") standardize_features = (value == "true");
            else if (key == "shuffle_data") shuffle_data = (value == "true");
            else if (key == "train_split_ratio") train_split_ratio = std::stod(value);
            else if (key == "validation_split_ratio") validation_split_ratio = std::stod(value);
            else if (key == "random_seed") random_seed = std::stoi(value);
            else if (key == "remove_outliers") remove_outliers = (value == "true");
            else if (key == "outlier_threshold") outlier_threshold = std::stod(value);
        }
    }
}

// DataLoader implementation
DataLoader::DataLoader(const PreprocessingOptions& options) : options_(options) {
    rng_.seed(options.random_seed);
}

std::vector<double> DataLoader::normalize_features(const std::vector<double>& features,
                                                  const std::vector<double>& mean,
                                                  const std::vector<double>& std_dev) {
    std::vector<double> normalized(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        normalized[i] = (features[i] - mean[i]) / std_dev[i];
    }
    return normalized;
}

std::vector<double> DataLoader::standardize_features(const std::vector<double>& features,
                                                    const std::vector<double>& mean,
                                                    const std::vector<double>& std_dev) {
    return normalize_features(features, mean, std_dev);
}

std::vector<DataPoint> DataLoader::remove_outliers(const std::vector<DataPoint>& data) {
    if (data.empty()) return data;
    
    std::vector<DataPoint> filtered_data;
    size_t num_features = data[0].features.size();
    
    // Calculate mean and standard deviation for each feature
    std::vector<double> mean(num_features, 0.0);
    std::vector<double> std_dev(num_features, 0.0);
    
    // Calculate mean
    for (const auto& data_point : data) {
        for (size_t i = 0; i < num_features; i++) {
            mean[i] += data_point.features[i];
        }
    }
    
    for (size_t i = 0; i < num_features; i++) {
        mean[i] /= data.size();
    }
    
    // Calculate standard deviation
    for (const auto& data_point : data) {
        for (size_t i = 0; i < num_features; i++) {
            double diff = data_point.features[i] - mean[i];
            std_dev[i] += diff * diff;
        }
    }
    
    for (size_t i = 0; i < num_features; i++) {
        std_dev[i] = std::sqrt(std_dev[i] / data.size());
    }
    
    // Filter outliers
    for (const auto& data_point : data) {
        bool is_outlier = false;
        for (size_t i = 0; i < num_features; i++) {
            double z_score = std::abs((data_point.features[i] - mean[i]) / std_dev[i]);
            if (z_score > options_.outlier_threshold) {
                is_outlier = true;
                break;
            }
        }
        
        if (!is_outlier) {
            filtered_data.push_back(data_point);
        }
    }
    
    return filtered_data;
}

std::vector<DataPoint> DataLoader::shuffle_data(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> shuffled = data;
    std::shuffle(shuffled.begin(), shuffled.end(), rng_);
    return shuffled;
}

std::vector<DataPoint> DataLoader::load_csv(const std::string& filename, 
                                           int label_column, 
                                           bool has_header) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::vector<DataPoint> data;
    std::string line;
    
    // Skip header if present
    if (has_header) {
        std::getline(file, line);
    }
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) continue;
        
        std::vector<double> features;
        int class_label = 0;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i == static_cast<size_t>(label_column)) {
                class_label = std::stoi(tokens[i]);
            } else {
                features.push_back(std::stod(tokens[i]));
            }
        }
        
        data.emplace_back(features, class_label);
    }
    
    return data;
}

std::vector<DataPoint> DataLoader::load_mnist(const std::string& images_file,
                                             const std::string& labels_file) {
    // This is a simplified MNIST loader
    // In practice, you would implement proper MNIST file format parsing
    std::vector<DataPoint> data;
    
    // For now, return empty data - MNIST loading is handled by MNISTLoader
    return data;
}

std::vector<DataPoint> DataLoader::load_iris(const std::string& filename) {
    // This is a simplified Iris loader
    // In practice, you would implement proper Iris file format parsing
    std::vector<DataPoint> data;
    
    // For now, return empty data - Iris loading is handled by IrisLoader
    return data;
}

std::vector<DataPoint> DataLoader::load_custom_format(const std::string& filename,
                                                     const std::string& format) {
    if (format == "csv") {
        return load_csv(filename);
    } else {
        throw std::invalid_argument("Unsupported format: " + format);
    }
}

DataLoader::DataSplit DataLoader::split_data(const std::vector<DataPoint>& data) {
    return split_data(data, options_.train_split_ratio, options_.validation_split_ratio);
}

DataLoader::DataSplit DataLoader::split_data(const std::vector<DataPoint>& data,
                                            double train_ratio,
                                            double validation_ratio) {
    if (train_ratio + validation_ratio > 1.0) {
        throw std::invalid_argument("Train ratio + validation ratio cannot exceed 1.0");
    }
    
    std::vector<DataPoint> shuffled_data = options_.shuffle_data ? shuffle_data(data) : data;
    
    DataSplit split;
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    size_t validation_size = static_cast<size_t>(data.size() * validation_ratio);
    
    split.train.assign(shuffled_data.begin(), shuffled_data.begin() + train_size);
    split.validation.assign(shuffled_data.begin() + train_size, 
                           shuffled_data.begin() + train_size + validation_size);
    split.test.assign(shuffled_data.begin() + train_size + validation_size, 
                     shuffled_data.end());
    
    return split;
}

std::vector<DataPoint> DataLoader::preprocess_data(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> processed = data;
    
    if (options_.remove_outliers) {
        processed = remove_outliers(processed);
    }
    
    if (options_.normalize_features) {
        processed = normalize_data(processed);
    } else if (options_.standardize_features) {
        processed = standardize_data(processed);
    }
    
    return processed;
}

std::vector<DataPoint> DataLoader::normalize_data(const std::vector<DataPoint>& data) {
    if (data.empty()) return data;
    
    std::vector<DataPoint> normalized = data;
    size_t num_features = data[0].features.size();
    
    // Calculate min and max for each feature
    std::vector<double> min_vals(num_features, std::numeric_limits<double>::max());
    std::vector<double> max_vals(num_features, std::numeric_limits<double>::lowest());
    
    for (const auto& data_point : data) {
        for (size_t i = 0; i < num_features; i++) {
            min_vals[i] = std::min(min_vals[i], data_point.features[i]);
            max_vals[i] = std::max(max_vals[i], data_point.features[i]);
        }
    }
    
    // Normalize features to [0, 1]
    for (auto& data_point : normalized) {
        for (size_t i = 0; i < num_features; i++) {
            if (max_vals[i] > min_vals[i]) {
                data_point.features[i] = (data_point.features[i] - min_vals[i]) / 
                                        (max_vals[i] - min_vals[i]);
            }
        }
    }
    
    return normalized;
}

std::vector<DataPoint> DataLoader::standardize_data(const std::vector<DataPoint>& data) {
    if (data.empty()) return data;
    
    std::vector<DataPoint> standardized = data;
    size_t num_features = data[0].features.size();
    
    // Calculate mean and standard deviation for each feature
    std::vector<double> mean(num_features, 0.0);
    std::vector<double> std_dev(num_features, 0.0);
    
    // Calculate mean
    for (const auto& data_point : data) {
        for (size_t i = 0; i < num_features; i++) {
            mean[i] += data_point.features[i];
        }
    }
    
    for (size_t i = 0; i < num_features; i++) {
        mean[i] /= data.size();
    }
    
    // Calculate standard deviation
    for (const auto& data_point : data) {
        for (size_t i = 0; i < num_features; i++) {
            double diff = data_point.features[i] - mean[i];
            std_dev[i] += diff * diff;
        }
    }
    
    for (size_t i = 0; i < num_features; i++) {
        std_dev[i] = std::sqrt(std_dev[i] / data.size());
    }
    
    // Standardize features
    for (auto& data_point : standardized) {
        for (size_t i = 0; i < num_features; i++) {
            if (std_dev[i] > 0) {
                data_point.features[i] = (data_point.features[i] - mean[i]) / std_dev[i];
            }
        }
    }
    
    return standardized;
}

DatasetInfo DataLoader::get_dataset_info(const std::vector<DataPoint>& data) const {
    DatasetInfo info;
    info.num_samples = data.size();
    
    if (data.empty()) return info;
    
    info.num_features = data[0].features.size();
    
    std::set<int> classes;
    std::map<int, int> class_counts;
    
    for (const auto& data_point : data) {
        classes.insert(data_point.class_label);
        class_counts[data_point.class_label]++;
    }
    
    info.num_classes = classes.size();
    
    for (const auto& [class_label, count] : class_counts) {
        info.class_distribution.push_back(count);
    }
    
    return info;
}

void DataLoader::print_data_statistics(const std::vector<DataPoint>& data) const {
    auto info = get_dataset_info(data);
    info.print_summary();
}

void DataLoader::save_data(const std::vector<DataPoint>& data, 
                          const std::string& filename,
                          const std::string& format) {
    if (format != "csv") {
        throw std::invalid_argument("Unsupported format: " + format);
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Write header
    if (!data.empty()) {
        for (size_t i = 0; i < data[0].features.size(); i++) {
            file << "feature_" << i;
            if (i < data[0].features.size() - 1) file << ",";
        }
        file << ",label" << std::endl;
    }
    
    // Write data
    for (const auto& data_point : data) {
        for (size_t i = 0; i < data_point.features.size(); i++) {
            file << data_point.features[i];
            if (i < data_point.features.size() - 1) file << ",";
        }
        file << "," << data_point.class_label << std::endl;
    }
}

bool DataLoader::validate_data(const std::vector<DataPoint>& data) const {
    if (data.empty()) return false;
    
    size_t expected_dim = data[0].features.size();
    std::set<int> classes;
    
    for (const auto& data_point : data) {
        if (data_point.features.size() != expected_dim) return false;
        classes.insert(data_point.class_label);
    }
    
    return classes.size() >= 2; // At least 2 classes needed for classification
}

std::vector<DataPoint> DataLoader::sample_data(const std::vector<DataPoint>& data, 
                                              int num_samples) {
    if (num_samples >= static_cast<int>(data.size())) {
        return data;
    }
    
    std::vector<DataPoint> shuffled = shuffle_data(data);
    return std::vector<DataPoint>(shuffled.begin(), shuffled.begin() + num_samples);
}

// MNISTLoader implementation
MNISTLoader::MNISTLoader(const std::string& data_dir) : data_dir_(data_dir) {}

std::vector<std::vector<double>> MNISTLoader::read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST images file: " + filename);
    }
    
    // Read magic number
    int magic_number = reverse_int(file.get() << 24 | file.get() << 16 | file.get() << 8 | file.get());
    if (magic_number != 0x803) {
        throw std::runtime_error("Invalid MNIST images file format");
    }
    
    // Read number of images
    int num_images = reverse_int(file.get() << 24 | file.get() << 16 | file.get() << 8 | file.get());
    
    // Read number of rows and columns
    int num_rows = reverse_int(file.get() << 24 | file.get() << 16 | file.get() << 8 | file.get());
    int num_cols = reverse_int(file.get() << 24 | file.get() << 16 | file.get() << 8 | file.get());
    
    std::vector<std::vector<double>> images;
    images.reserve(num_images);
    
    for (int i = 0; i < num_images; i++) {
        std::vector<double> image(num_rows * num_cols);
        for (int j = 0; j < num_rows * num_cols; j++) {
            unsigned char pixel = file.get();
            image[j] = static_cast<double>(pixel) / 255.0; // Normalize to [0, 1]
        }
        images.push_back(image);
    }
    
    return images;
}

std::vector<int> MNISTLoader::read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST labels file: " + filename);
    }
    
    // Read magic number
    int magic_number = reverse_int(file.get() << 24 | file.get() << 16 | file.get() << 8 | file.get());
    if (magic_number != 0x801) {
        throw std::runtime_error("Invalid MNIST labels file format");
    }
    
    // Read number of labels
    int num_labels = reverse_int(file.get() << 24 | file.get() << 16 | file.get() << 8 | file.get());
    
    std::vector<int> labels;
    labels.reserve(num_labels);
    
    for (int i = 0; i < num_labels; i++) {
        unsigned char label = file.get();
        labels.push_back(static_cast<int>(label));
    }
    
    return labels;
}

int MNISTLoader::reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<DataPoint> MNISTLoader::load_training_data() {
    std::string images_file = data_dir_ + "/train-images-idx3-ubyte";
    std::string labels_file = data_dir_ + "/train-labels-idx1-ubyte";
    
    auto images = read_mnist_images(images_file);
    auto labels = read_mnist_labels(labels_file);
    
    std::vector<DataPoint> data;
    data.reserve(images.size());
    
    for (size_t i = 0; i < images.size(); i++) {
        data.emplace_back(images[i], labels[i]);
    }
    
    return data;
}

std::vector<DataPoint> MNISTLoader::load_test_data() {
    std::string images_file = data_dir_ + "/t10k-images-idx3-ubyte";
    std::string labels_file = data_dir_ + "/t10k-labels-idx1-ubyte";
    
    auto images = read_mnist_images(images_file);
    auto labels = read_mnist_labels(labels_file);
    
    std::vector<DataPoint> data;
    data.reserve(images.size());
    
    for (size_t i = 0; i < images.size(); i++) {
        data.emplace_back(images[i], labels[i]);
    }
    
    return data;
}

std::vector<DataPoint> MNISTLoader::load_full_dataset() {
    auto train_data = load_training_data();
    auto test_data = load_test_data();
    
    train_data.insert(train_data.end(), test_data.begin(), test_data.end());
    return train_data;
}

std::vector<DataPoint> MNISTLoader::preprocess_mnist(const std::vector<DataPoint>& data) {
    return preprocess_data(data);
}

std::vector<double> MNISTLoader::flatten_image(const std::vector<std::vector<double>>& image) {
    if (image.empty()) return {};
    
    std::vector<double> flattened;
    for (const auto& row : image) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    
    return flattened;
}

// IrisLoader implementation
void IrisLoader::initialize_class_mapping() {
    class_mapping_["setosa"] = 0;
    class_mapping_["versicolor"] = 1;
    class_mapping_["virginica"] = 2;
}

std::vector<DataPoint> IrisLoader::load_iris_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open Iris file: " + filename);
    }
    
    std::vector<DataPoint> data;
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 5) { // sepal_length, sepal_width, petal_length, petal_width, species
            std::vector<double> features;
            features.push_back(std::stod(tokens[0])); // sepal_length
            features.push_back(std::stod(tokens[1])); // sepal_width
            features.push_back(std::stod(tokens[2])); // petal_length
            features.push_back(std::stod(tokens[3])); // petal_width
            
            // Convert species name to class ID
            std::string species = tokens[4];
            int class_id = get_class_id(species);
            if (class_id == -1) {
                // Add new class if not found
                class_id = class_mapping_.size();
                class_mapping_[species] = class_id;
            }
            
            data.emplace_back(features, class_id);
        }
    }
    
    return data;
}

std::vector<DataPoint> IrisLoader::load_iris_builtin() {
    // Built-in Iris dataset (Fisher's original data)
    std::vector<DataPoint> data;
    
    // Setosa samples
    data.emplace_back(std::vector<double>{5.1, 3.5, 1.4, 0.2}, 0);
    data.emplace_back(std::vector<double>{4.9, 3.0, 1.4, 0.2}, 0);
    data.emplace_back(std::vector<double>{4.7, 3.2, 1.3, 0.2}, 0);
    data.emplace_back(std::vector<double>{4.6, 3.1, 1.5, 0.2}, 0);
    data.emplace_back(std::vector<double>{5.0, 3.6, 1.4, 0.2}, 0);
    
    // Versicolor samples
    data.emplace_back(std::vector<double>{7.0, 3.2, 4.7, 1.4}, 1);
    data.emplace_back(std::vector<double>{6.4, 3.2, 4.5, 1.5}, 1);
    data.emplace_back(std::vector<double>{6.9, 3.1, 4.9, 1.5}, 1);
    data.emplace_back(std::vector<double>{5.5, 2.3, 4.0, 1.3}, 1);
    data.emplace_back(std::vector<double>{6.5, 2.8, 4.6, 1.5}, 1);
    
    // Virginica samples
    data.emplace_back(std::vector<double>{6.3, 3.3, 6.0, 2.5}, 2);
    data.emplace_back(std::vector<double>{5.8, 2.7, 5.1, 1.9}, 2);
    data.emplace_back(std::vector<double>{7.1, 3.0, 5.9, 2.1}, 2);
    data.emplace_back(std::vector<double>{6.3, 2.9, 5.6, 1.8}, 2);
    data.emplace_back(std::vector<double>{6.5, 3.0, 5.8, 2.2}, 2);
    
    return data;
}

std::vector<DataPoint> IrisLoader::load_full_dataset() {
    if (!data_file_.empty() && file_utils::file_exists(data_file_)) {
        return load_iris_from_file(data_file_);
    } else {
        // Try to load from default location
        std::string default_file = "data/iris.csv";
        if (file_utils::file_exists(default_file)) {
            return load_iris_from_file(default_file);
        } else {
            // Fall back to built-in data
            return load_iris_builtin();
        }
    }
}

std::vector<DataPoint> IrisLoader::load_training_data() {
    auto full_data = load_full_dataset();
    auto split = split_data(full_data);
    return split.train;
}

std::vector<DataPoint> IrisLoader::load_test_data() {
    auto full_data = load_full_dataset();
    auto split = split_data(full_data);
    return split.test;
}

std::vector<DataPoint> IrisLoader::preprocess_iris(const std::vector<DataPoint>& data) {
    return preprocess_data(data);
}

std::string IrisLoader::get_class_name(int class_id) const {
    // First check if we have a mapping for this class_id
    for (const auto& [name, id] : class_mapping_) {
        if (id == class_id) {
            return name;
        }
    }
    
    // Fall back to default names if no mapping found
    switch (class_id) {
        case 0: return "setosa";
        case 1: return "versicolor";
        case 2: return "virginica";
        default: return "unknown";
    }
}

int IrisLoader::get_class_id(const std::string& class_name) const {
    auto it = class_mapping_.find(class_name);
    if (it != class_mapping_.end()) {
        return it->second;
    }
    return -1;
}

} // namespace lvq 