cmake_minimum_required(VERSION 3.10)
project(dnn_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find GTest
find_package(GTest REQUIRED)

# Source files
set(SOURCES
    src/nn/activation.cpp
    src/nn/layer.cpp
    src/nn/loss.cpp
    src/nn/model.cpp
    src/optim/sgd.cpp
)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Create a library target
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Link against Eigen
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# Test executable
enable_testing()

add_executable(
  dnn_tests
  dnn_tests/activation_test.cpp
  dnn_tests/build_test.cpp
  src/nn/activation.cpp
)


target_link_libraries(
    dnn_tests
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(dnn_tests)
