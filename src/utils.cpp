#include "utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lvq {

// Global logger instance
log_utils::Logger log_utils::global_logger;

// Timer implementation
Timer::Timer() : is_running_(false) {}
void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}
void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = false;
}
double Timer::elapsed_seconds() const {
    auto end = is_running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double>(end - start_time_).count();
}
double Timer::elapsed_milliseconds() const {
    return elapsed_seconds() * 1000.0;
}

// ProgressBar implementation
ProgressBar::ProgressBar(int total, int width, const std::string& prefix)
    : total_(total), current_(0), width_(width), prefix_(prefix) {}
void ProgressBar::update(int current) {
    current_ = current;
    display();
}
void ProgressBar::increment() {
    update(current_ + 1);
}
void ProgressBar::display() const {
    double progress = static_cast<double>(current_) / total_;
    int pos = static_cast<int>(width_ * progress);
    std::cout << "\r" << prefix_ << " [";
    for (int i = 0; i < width_; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%";
    std::cout.flush();
}
void ProgressBar::finish() const {
    std::cout << std::endl;
}

namespace log_utils {
Logger::Logger() : log_level_(LogLevel::INFO), output_stream_(&std::cout) {}
Logger::Logger(LogLevel level) : log_level_(level), output_stream_(&std::cout) {}
Logger::Logger(LogLevel level, std::ostream& output)
    : log_level_(level), output_stream_(&output) {}
void Logger::set_level(LogLevel level) {
    log_level_ = level;
}
void Logger::set_output(std::ostream& output) {
    output_stream_ = &output;
}
void Logger::log(LogLevel level, const std::string& message) {
    if (level >= log_level_) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        *output_stream_ << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << " ";
        switch (level) {
            case LogLevel::DEBUG: *output_stream_ << "[DEBUG] "; break;
            case LogLevel::INFO: *output_stream_ << "[INFO] "; break;
            case LogLevel::WARNING: *output_stream_ << "[WARNING] "; break;
            case LogLevel::ERROR: *output_stream_ << "[ERROR] "; break;
        }
        *output_stream_ << message << std::endl;
    }
}
void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}
void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}
void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}
void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

// Global logging functions
void set_log_level(LogLevel level) {
    global_logger.set_level(level);
}

void log_debug(const std::string& message) {
    global_logger.debug(message);
}

void log_info(const std::string& message) {
    global_logger.info(message);
}

void log_warning(const std::string& message) {
    global_logger.warning(message);
}

void log_error(const std::string& message) {
    global_logger.error(message);
}
} // namespace log_utils

// ConfigManager implementation
ConfigManager::ConfigManager() {}
ConfigManager::ConfigManager(const std::string& filename) {
    load_config(filename);
}
void ConfigManager::load_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            config_[key] = value;
        }
    }
}
void ConfigManager::save_config(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file for writing: " + filename);
    }
    for (const auto& [key, value] : config_) {
        file << key << "=" << value << std::endl;
    }
}
std::string ConfigManager::get_string(const std::string& key, const std::string& default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) return it->second;
    return default_value;
}
int ConfigManager::get_int(const std::string& key, int default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) return std::stoi(it->second);
    return default_value;
}
double ConfigManager::get_double(const std::string& key, double default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) return std::stod(it->second);
    return default_value;
}
bool ConfigManager::get_bool(const std::string& key, bool default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) return it->second == "true";
    return default_value;
}
void ConfigManager::set_string(const std::string& key, const std::string& value) {
    config_[key] = value;
}
void ConfigManager::set_int(const std::string& key, int value) {
    config_[key] = std::to_string(value);
}
void ConfigManager::set_double(const std::string& key, double value) {
    config_[key] = std::to_string(value);
}
void ConfigManager::set_bool(const std::string& key, bool value) {
    config_[key] = value ? "true" : "false";
}
bool ConfigManager::has_key(const std::string& key) const {
    return config_.find(key) != config_.end();
}
void ConfigManager::remove_key(const std::string& key) {
    config_.erase(key);
}
void ConfigManager::clear() {
    config_.clear();
}
std::vector<std::string> ConfigManager::get_keys() const {
    std::vector<std::string> keys;
    for (const auto& [key, _] : config_) {
        keys.push_back(key);
    }
    return keys;
}
void ConfigManager::print_config() const {
    for (const auto& [key, value] : config_) {
        std::cout << key << " = " << value << std::endl;
    }
}

// Math utilities implementation
double math_utils::euclidean_distance(const std::vector<double>& v1, 
                                     const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

double math_utils::manhattan_distance(const std::vector<double>& v1, 
                                     const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        sum += std::abs(v1[i] - v2[i]);
    }
    
    return sum;
}

double math_utils::cosine_similarity(const std::vector<double>& v1, 
                                    const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }
    
    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (size_t i = 0; i < v1.size(); i++) {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }
    
    return dot_product / (norm1 * norm2);
}

std::pair<double, double> math_utils::minmax(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Vector is empty");
    }
    
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    return {*min_it, *max_it};
}

double math_utils::mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Vector is empty");
    }
    
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double math_utils::standard_deviation(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Vector is empty");
    }
    
    double avg = mean(values);
    double sum_sq_diff = 0.0;
    
    for (double value : values) {
        double diff = value - avg;
        sum_sq_diff += diff * diff;
    }
    
    return std::sqrt(sum_sq_diff / values.size());
}

std::vector<double> math_utils::normalize(const std::vector<double>& values) {
    if (values.empty()) {
        return values;
    }
    
    auto [min_val, max_val] = minmax(values);
    if (max_val == min_val) {
        return std::vector<double>(values.size(), 0.5);
    }
    
    std::vector<double> normalized(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        normalized[i] = (values[i] - min_val) / (max_val - min_val);
    }
    
    return normalized;
}

std::vector<double> math_utils::standardize(const std::vector<double>& values) {
    if (values.empty()) {
        return values;
    }
    
    double avg = mean(values);
    double std_dev = standard_deviation(values);
    
    if (std_dev == 0.0) {
        return std::vector<double>(values.size(), 0.0);
    }
    
    std::vector<double> standardized(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        standardized[i] = (values[i] - avg) / std_dev;
    }
    
    return standardized;
}

// File utilities implementation
bool file_utils::file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

bool file_utils::create_directory(const std::string& dirname) {
    // Simple implementation - in a real project you'd use filesystem
    std::string cmd = "mkdir -p " + dirname;
    return system(cmd.c_str()) == 0;
}

bool file_utils::directory_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

std::string file_utils::get_file_extension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
        return filename.substr(pos + 1);
    }
    return "";
}

std::string file_utils::get_filename_without_extension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
        return filename.substr(0, pos);
    }
    return filename;
}

std::string file_utils::get_directory(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(0, pos);
    }
    return "";
}

std::vector<std::string> file_utils::list_files(const std::string& directory, 
                                               const std::string& extension) {
    // Simple implementation - in a real project you'd use filesystem
    std::vector<std::string> files;
    std::string cmd = "ls " + directory;
    if (!extension.empty()) {
        cmd += "/*." + extension;
    }
    // This is a simplified implementation
    return files;
}

size_t file_utils::get_file_size(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return 0;
    }
    return file.tellg();
}

std::vector<std::string> file_utils::read_lines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    
    return lines;
}

void file_utils::write_lines(const std::string& filename, 
                            const std::vector<std::string>& lines) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    for (const auto& line : lines) {
        file << line << std::endl;
    }
}

// String utilities implementation
std::vector<std::string> string_utils::split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string string_utils::join(const std::vector<std::string>& strings, 
                               const std::string& delimiter) {
    if (strings.empty()) {
        return "";
    }
    
    std::string result = strings[0];
    for (size_t i = 1; i < strings.size(); i++) {
        result += delimiter + strings[i];
    }
    
    return result;
}

std::string string_utils::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::string string_utils::to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string string_utils::to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

bool string_utils::starts_with(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool string_utils::ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

std::string string_utils::replace(const std::string& str, const std::string& from, 
                                 const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

// Random utilities implementation
namespace random_utils {
    static std::mt19937 rng_(std::random_device{}());
    
    void set_seed(int seed) {
        rng_.seed(seed);
    }
    
    int get_random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }
    
    double get_random_double(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(rng_);
    }
    
    std::vector<double> get_random_vector(int size, double min, double max) {
        std::vector<double> vec(size);
        for (int i = 0; i < size; i++) {
            vec[i] = get_random_double(min, max);
        }
        return vec;
    }
    
    std::vector<int> get_random_permutation(int size) {
        std::vector<int> perm(size);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);
        return perm;
    }
}

// Config utilities implementation
namespace config_utils {
    std::map<std::string, std::string> load_config(const std::string& filename) {
        std::map<std::string, std::string> config;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + filename);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            line = string_utils::trim(line);
            
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = string_utils::trim(line.substr(0, pos));
                std::string value = string_utils::trim(line.substr(pos + 1));
                config[key] = value;
            }
        }
        
        return config;
    }
    
    void save_config(const std::map<std::string, std::string>& config, 
                    const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file for writing: " + filename);
        }
        
        for (const auto& [key, value] : config) {
            file << key << "=" << value << std::endl;
        }
    }
    
    std::string get_config_value(const std::map<std::string, std::string>& config, 
                                const std::string& key, 
                                const std::string& default_value) {
        auto it = config.find(key);
        return (it != config.end()) ? it->second : default_value;
    }
    
    int get_config_int(const std::map<std::string, std::string>& config, 
                      const std::string& key, int default_value) {
        auto it = config.find(key);
        if (it != config.end()) {
            try {
                return std::stoi(it->second);
            } catch (const std::exception&) {
                return default_value;
            }
        }
        return default_value;
    }
    
    double get_config_double(const std::map<std::string, std::string>& config, 
                            const std::string& key, double default_value) {
        auto it = config.find(key);
        if (it != config.end()) {
            try {
                return std::stod(it->second);
            } catch (const std::exception&) {
                return default_value;
            }
        }
        return default_value;
    }
    
    bool get_config_bool(const std::map<std::string, std::string>& config, 
                        const std::string& key, bool default_value) {
        auto it = config.find(key);
        if (it != config.end()) {
            std::string value = string_utils::to_lower(it->second);
            return (value == "true" || value == "1" || value == "yes");
        }
        return default_value;
    }
}

} // namespace lvq 