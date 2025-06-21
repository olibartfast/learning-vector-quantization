#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "../include/lvq_network.h"

namespace py = pybind11;
using namespace lvq;

// Helper function to convert numpy array to vector
std::vector<double> numpy_to_vector(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    std::vector<double> result(buf.shape[0]);
    for (size_t i = 0; i < buf.shape[0]; i++) {
        result[i] = static_cast<double*>(buf.ptr)[i];
    }
    return result;
}

// Helper function to convert vector to numpy array
py::array_t<double> vector_to_numpy(const std::vector<double>& vec) {
    py::array_t<double> result(vec.size());
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i];
    }
    return result;
}

// Helper function to convert list of vectors to vector of vectors
std::vector<std::vector<double>> list_to_vector_of_vectors(py::list lst) {
    std::vector<std::vector<double>> result;
    for (py::handle item : lst) {
        if (py::isinstance<py::array_t<double>>(item)) {
            result.push_back(numpy_to_vector(item.cast<py::array_t<double>>()));
        } else if (py::isinstance<py::list>(item)) {
            py::list sublist = item.cast<py::list>();
            std::vector<double> vec;
            for (py::handle subitem : sublist) {
                vec.push_back(subitem.cast<double>());
            }
            result.push_back(vec);
        }
    }
    return result;
}

PYBIND11_MODULE(lvq_python, m) {
    m.doc() = "Python bindings for Learning Vector Quantization (LVQ) Network"; // optional module docstring

    // Bind DataPoint class
    py::class_<DataPoint>(m, "DataPoint")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, int>())
        .def_readwrite("features", &DataPoint::features)
        .def_readwrite("class_label", &DataPoint::class_label)
        .def("get_dimension", &DataPoint::get_dimension)
        .def("__repr__", [](const DataPoint& dp) {
            return "DataPoint(features=" + std::to_string(dp.features.size()) + 
                   " features, class_label=" + std::to_string(dp.class_label) + ")";
        });

    // Bind CodebookVector class
    py::class_<CodebookVector>(m, "CodebookVector")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, int>())
        .def_readwrite("weights", &CodebookVector::weights)
        .def_readwrite("class_label", &CodebookVector::class_label)
        .def_readwrite("frequency", &CodebookVector::frequency)
        .def("get_dimension", &CodebookVector::get_dimension)
        .def("__repr__", [](const CodebookVector& cv) {
            return "CodebookVector(weights=" + std::to_string(cv.weights.size()) + 
                   " weights, class_label=" + std::to_string(cv.class_label) + 
                   ", frequency=" + std::to_string(cv.frequency) + ")";
        });

    // Bind LVQConfig struct
    py::class_<LVQConfig>(m, "LVQConfig")
        .def(py::init<>())
        .def_readwrite("num_codebook_vectors", &LVQConfig::num_codebook_vectors)
        .def_readwrite("learning_rate", &LVQConfig::learning_rate)
        .def_readwrite("learning_rate_decay", &LVQConfig::learning_rate_decay)
        .def_readwrite("max_iterations", &LVQConfig::max_iterations)
        .def_readwrite("convergence_threshold", &LVQConfig::convergence_threshold)
        .def_readwrite("use_adaptive_lr", &LVQConfig::use_adaptive_lr)
        .def_readwrite("distance_metric", &LVQConfig::distance_metric)
        .def_readwrite("random_seed", &LVQConfig::random_seed)
        .def("save", &LVQConfig::save)
        .def("load", &LVQConfig::load)
        .def("__repr__", [](const LVQConfig& config) {
            return "LVQConfig(num_codebook_vectors=" + std::to_string(config.num_codebook_vectors) + 
                   ", learning_rate=" + std::to_string(config.learning_rate) + 
                   ", max_iterations=" + std::to_string(config.max_iterations) + ")";
        });

    // Bind LVQNetwork class
    py::class_<LVQNetwork>(m, "LVQNetwork")
        .def(py::init<>())
        .def(py::init<const LVQConfig&>())
        
        // Training methods
        .def("train", [](LVQNetwork& self, const std::vector<DataPoint>& training_data) {
            self.train(training_data);
        })
        .def("train_online", &LVQNetwork::train_online)
        
        // Prediction methods with numpy support
        .def("predict", [](const LVQNetwork& self, py::array_t<double> features) {
            return self.predict(numpy_to_vector(features));
        })
        .def("predict_batch", [](const LVQNetwork& self, py::list features_list) {
            auto features = list_to_vector_of_vectors(features_list);
            return self.predict_batch(features);
        })
        .def("predict_with_confidence", [](const LVQNetwork& self, py::array_t<double> features) {
            auto result = self.predict_with_confidence(numpy_to_vector(features));
            return py::make_tuple(result.first, result.second);
        })
        
        // Model persistence
        .def("save_model", &LVQNetwork::save_model)
        .def("load_model", &LVQNetwork::load_model)
        
        // Getters
        .def("get_codebook_vectors", &LVQNetwork::get_codebook_vectors)
        .def("get_config", &LVQNetwork::get_config)
        .def("is_trained", &LVQNetwork::is_trained)
        .def("get_unique_classes", &LVQNetwork::get_unique_classes)
        .def("get_input_dimension", &LVQNetwork::get_input_dimension)
        
        // Setters
        .def("set_config", &LVQNetwork::set_config)
        
        // Utility methods
        .def("reset", &LVQNetwork::reset)
        .def("print_summary", &LVQNetwork::print_summary)
        .def("get_decision_boundaries", &LVQNetwork::get_decision_boundaries)
        
        .def("__repr__", [](const LVQNetwork& self) {
            return "LVQNetwork(trained=" + std::string(self.is_trained() ? "True" : "False") + 
                   ", input_dim=" + std::to_string(self.get_input_dimension()) + 
                   ", codebook_size=" + std::to_string(self.get_codebook_vectors().size()) + ")";
        });

    // Add convenience functions
    m.def("create_data_point", [](py::array_t<double> features, int class_label) {
        return DataPoint(numpy_to_vector(features), class_label);
    });

    m.def("array_to_data_points", [](py::array_t<double> features, py::array_t<int> labels) {
        py::buffer_info feat_buf = features.request();
        py::buffer_info label_buf = labels.request();
        
        if (feat_buf.ndim != 2) {
            throw std::runtime_error("Features must be a 2D array");
        }
        if (label_buf.ndim != 1) {
            throw std::runtime_error("Labels must be a 1D array");
        }
        if (feat_buf.shape[0] != label_buf.shape[0]) {
            throw std::runtime_error("Number of samples must match number of labels");
        }
        
        std::vector<DataPoint> data_points;
        size_t num_samples = feat_buf.shape[0];
        size_t num_features = feat_buf.shape[1];
        
        double* feat_ptr = static_cast<double*>(feat_buf.ptr);
        int* label_ptr = static_cast<int*>(label_buf.ptr);
        
        for (size_t i = 0; i < num_samples; i++) {
            std::vector<double> sample_features(num_features);
            for (size_t j = 0; j < num_features; j++) {
                sample_features[j] = feat_ptr[i * num_features + j];
            }
            data_points.emplace_back(sample_features, label_ptr[i]);
        }
        
        return data_points;
    });

    m.def("data_points_to_arrays", [](const std::vector<DataPoint>& data_points) {
        if (data_points.empty()) {
            throw std::runtime_error("Data points list is empty");
        }
        
        size_t num_samples = data_points.size();
        size_t num_features = data_points[0].features.size();
        
        py::array_t<double> features({num_samples, num_features});
        py::array_t<int> labels(num_samples);
        
        py::buffer_info feat_buf = features.request();
        py::buffer_info label_buf = labels.request();
        
        double* feat_ptr = static_cast<double*>(feat_buf.ptr);
        int* label_ptr = static_cast<int*>(label_buf.ptr);
        
        for (size_t i = 0; i < num_samples; i++) {
            if (data_points[i].features.size() != num_features) {
                throw std::runtime_error("All data points must have the same number of features");
            }
            
            for (size_t j = 0; j < num_features; j++) {
                feat_ptr[i * num_features + j] = data_points[i].features[j];
            }
            label_ptr[i] = data_points[i].class_label;
        }
        
        return py::make_tuple(features, labels);
    });
} 