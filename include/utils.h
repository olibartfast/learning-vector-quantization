#pragma once

#include "lvq_network.h"
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lvq {

/**
 * @brief Timer utility for performance measurement
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_;
public:
    Timer();
    void start();
    void stop();
    double elapsed_seconds() const;
    double elapsed_milliseconds() const;
};

/**
 * @brief Progress bar utility
 */
class ProgressBar {
private:
    int total_;
    int current_;
    int width_;
    std::string prefix_;
public:
    ProgressBar(int total, int width = 50, const std::string& prefix = "Progress");
    void update(int current);
    void increment();
    void display() const;
    void finish() const;
};

/**
 * @brief File utilities
 */
namespace file_utils {
    bool file_exists(const std::string& filename);
    std::string get_file_extension(const std::string& filename);
    std::string get_filename_without_extension(const std::string& filename);
    std::string get_directory(const std::string& filepath);
    bool create_directory(const std::string& dirname);
    bool directory_exists(const std::string& path);
    std::vector<std::string> list_files(const std::string& directory, 
                                       const std::string& extension = "");
    size_t get_file_size(const std::string& filename);
    std::vector<std::string> read_lines(const std::string& filename);
    void write_lines(const std::string& filename, const std::vector<std::string>& lines);
}

/**
 * @brief String utilities
 */
namespace string_utils {
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string trim(const std::string& str);
    std::string to_lower(const std::string& str);
    std::string to_upper(const std::string& str);
    bool starts_with(const std::string& str, const std::string& prefix);
    bool ends_with(const std::string& str, const std::string& suffix);
    std::string replace(const std::string& str, const std::string& from, 
                       const std::string& to);
    std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
}

/**
 * @brief Math utilities
 */
namespace math_utils {
    double euclidean_distance(const std::vector<double>& v1, 
                             const std::vector<double>& v2);
    double manhattan_distance(const std::vector<double>& v1, 
                             const std::vector<double>& v2);
    double cosine_similarity(const std::vector<double>& v1, 
                            const std::vector<double>& v2);
    double mean(const std::vector<double>& values);
    double std_dev(const std::vector<double>& values);
    double variance(const std::vector<double>& values);
    double standard_deviation(const std::vector<double>& values);
    std::vector<double> normalize_vector(const std::vector<double>& vec);
    std::vector<double> standardize_vector(const std::vector<double>& vec);
    std::vector<double> normalize(const std::vector<double>& values);
    std::vector<double> standardize(const std::vector<double>& values);
    double min(const std::vector<double>& values);
    double max(const std::vector<double>& values);
    std::pair<double, double> minmax(const std::vector<double>& values);
}

/**
 * @brief Random utilities
 */
namespace random_utils {
    void set_seed(int seed);
    int get_random_int(int min, int max);
    double get_random_double(double min, double max);
    std::vector<double> get_random_vector(int size, double min, double max);
    std::vector<int> get_random_permutation(int size);
    template<typename T>
    std::vector<T> shuffle_vector(const std::vector<T>& vec);
}

/**
 * @brief Statistics utilities
 */
namespace stats_utils {
    double accuracy(const std::vector<int>& predictions, 
                   const std::vector<int>& true_labels);
    double precision(const std::vector<int>& predictions, 
                    const std::vector<int>& true_labels, int class_label);
    double recall(const std::vector<int>& predictions, 
                 const std::vector<int>& true_labels, int class_label);
    double f1_score(const std::vector<int>& predictions, 
                   const std::vector<int>& true_labels, int class_label);
    std::vector<double> confusion_matrix(const std::vector<int>& predictions, 
                                        const std::vector<int>& true_labels, 
                                        int num_classes);
}

/**
 * @brief Visualization utilities
 */
namespace viz_utils {
    void print_vector(const std::vector<double>& vec, 
                     const std::string& name = "Vector");
    void print_matrix(const std::vector<std::vector<double>>& matrix, 
                     const std::string& name = "Matrix");
    void print_confusion_matrix(const std::vector<std::vector<int>>& cm, 
                               const std::vector<std::string>& class_names = {});
    void save_plot_data(const std::vector<double>& x, 
                       const std::vector<double>& y, 
                       const std::string& filename);
}

/**
 * @brief Configuration utilities
 */
namespace config_utils {
    struct ConfigValue {
        std::string key;
        std::string value;
        std::string type;
    };
    
    std::map<std::string, std::string> load_config(const std::string& filename);
    void save_config(const std::map<std::string, std::string>& config, 
                    const std::string& filename);
    std::string get_config_value(const std::map<std::string, std::string>& config, 
                                const std::string& key, 
                                const std::string& default_value = "");
    int get_config_int(const std::map<std::string, std::string>& config, 
                      const std::string& key, int default_value = 0);
    double get_config_double(const std::map<std::string, std::string>& config, 
                            const std::string& key, double default_value = 0.0);
    bool get_config_bool(const std::map<std::string, std::string>& config, 
                        const std::string& key, bool default_value = false);
}

/**
 * @brief Logging utilities
 */
namespace log_utils {
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    class Logger {
    private:
        LogLevel log_level_;
        std::ostream* output_stream_;
    public:
        Logger();
        Logger(LogLevel level);
        Logger(LogLevel level, std::ostream& output);
        void set_level(LogLevel level);
        void set_output(std::ostream& output);
        void log(LogLevel level, const std::string& message);
        void debug(const std::string& message);
        void info(const std::string& message);
        void warning(const std::string& message);
        void error(const std::string& message);
    };

    // Global logger instance
    extern Logger global_logger;
    
    // Global logging functions
    void set_log_level(LogLevel level);
    void log_debug(const std::string& message);
    void log_info(const std::string& message);
    void log_warning(const std::string& message);
    void log_error(const std::string& message);
}

// ConfigManager declaration (add to lvq namespace)
class ConfigManager {
private:
    std::map<std::string, std::string> config_;
public:
    ConfigManager();
    explicit ConfigManager(const std::string& filename);
    void load_config(const std::string& filename);
    void save_config(const std::string& filename) const;
    std::string get_string(const std::string& key, const std::string& default_value = "") const;
    int get_int(const std::string& key, int default_value = 0) const;
    double get_double(const std::string& key, double default_value = 0.0) const;
    bool get_bool(const std::string& key, bool default_value = false) const;
    void set_string(const std::string& key, const std::string& value);
    void set_int(const std::string& key, int value);
    void set_double(const std::string& key, double value);
    void set_bool(const std::string& key, bool value);
    bool has_key(const std::string& key) const;
    void remove_key(const std::string& key);
    void clear();
    std::vector<std::string> get_keys() const;
    void print_config() const;
};

} // namespace lvq

// Template implementations
template<typename T>
std::vector<T> lvq::random_utils::shuffle_vector(const std::vector<T>& vec) {
    std::vector<T> result = vec;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(result.begin(), result.end(), g);
    return result;
} 