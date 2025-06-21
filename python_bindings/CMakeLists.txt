cmake_minimum_required(VERSION 3.10)
project(LVQ_Python_Bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(../include)

# Source files for the library
set(LIB_SOURCES
    ../src/lvq_network.cpp
    ../src/lvq_trainer.cpp
    ../src/lvq_tester.cpp
    ../src/data_loader.cpp
    ../src/utils.cpp
)

# Create the Python module
pybind11_add_module(lvq_python lvq_python.cpp ${LIB_SOURCES})

# Set compiler flags
target_compile_options(lvq_python PRIVATE -Wall -Wextra -O2)

# Link against pybind11
target_link_libraries(lvq_python PRIVATE pybind11::module)

# Set output directory
set_target_properties(lvq_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build
)

# Install the Python module
install(TARGETS lvq_python
        LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
        RUNTIME DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages) 