cmake_minimum_required(VERSION 3.10)
project(LVQ_Network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/lvq_network.cpp
    src/lvq_trainer.cpp
    src/lvq_tester.cpp
    src/data_loader.cpp
    src/utils.cpp
)

# Header files
set(HEADERS
    include/lvq_network.h
    include/lvq_trainer.h
    include/lvq_tester.h
    include/data_loader.h
    include/utils.h
)

# Create library
add_library(lvq_lib STATIC ${SOURCES} ${HEADERS})

# Create executables
add_executable(lvq_train src/main_train.cpp)
add_executable(lvq_test src/main_test.cpp)
add_executable(lvq_demo src/main_demo.cpp)
add_executable(lvq_iris_demo src/main_iris_demo.cpp)

# Link libraries
target_link_libraries(lvq_train lvq_lib)
target_link_libraries(lvq_test lvq_lib)
target_link_libraries(lvq_demo lvq_lib)
target_link_libraries(lvq_iris_demo lvq_lib)

# Install targets
install(TARGETS lvq_lib lvq_train lvq_test lvq_demo lvq_iris_demo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES ${HEADERS} DESTINATION include/lvq) 