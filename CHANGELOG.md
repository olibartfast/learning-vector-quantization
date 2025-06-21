# Changelog

## [2.0.0] - 2024-12-19

### Added
- **Complete project restructuring** with modern C++17 architecture
- **Modular design** with separate training, testing, and data loading modules
- **MNIST dataset support** with built-in loader and preprocessing
- **Iris dataset demo** with complete interactive classification example
- **Production-ready features**:
  - Comprehensive error handling and logging
  - Configuration management system
  - Model persistence (save/load)
  - Performance monitoring and timing
  - Progress tracking and callbacks
- **Multiple executables**:
  - `lvq_train`: Training tool with validation
  - `lvq_test`: Testing and evaluation tool
  - `lvq_demo`: Generic demo tool
  - `lvq_iris_demo`: Specialized Iris dataset demo
- **Advanced evaluation metrics**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix generation
  - Per-class metrics
  - Confidence scores
  - Top-k predictions
- **Build system**:
  - CMake-based build configuration
  - Automated build script (`build.sh`)
  - Cross-platform compatibility
- **Documentation**:
  - Comprehensive README with usage examples
  - API documentation
  - Performance benchmarks
  - Configuration examples

### Changed
- **Architecture**: From single-file implementation to modular, production-ready system
- **Language**: Modern C++17 with smart pointers and RAII
- **Design**: Generic implementation supporting any classification dataset
- **Interface**: Command-line tools with comprehensive options
- **Data handling**: Robust data loading and preprocessing pipeline

### Removed
- **Legacy files**:
  - `lvq.h`, `lvq_init.h`, `lvq_struct.h`, `lvq_rout.h`
  - `TestLvq.cpp`
  - `InTrain.txt`, `stat*.txt`, `o.txt`, `out_temp.txt`, `refvec.txt`, `blacklist.txt`
- **RFID-specific code**: Replaced with generic classification framework

### Technical Improvements
- **Memory management**: Smart pointers and RAII for automatic resource management
- **Error handling**: Comprehensive exception handling and validation
- **Performance**: Optimized algorithms and efficient data structures
- **Extensibility**: Easy to add new datasets and distance metrics
- **Testing**: Separate training and testing modules as requested
- **Usability**: Interactive demos and comprehensive help systems

## [1.0.0] - 2007

### Original Implementation
- Basic LVQ implementation for RFID positioning system
- Single-file C++ implementation
- Hardcoded parameters and limited flexibility
- Bachelor's thesis project 