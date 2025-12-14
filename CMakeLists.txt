cmake_minimum_required(VERSION 3.16)
project(smpl_ceres_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(ZLIB REQUIRED)

# Build cnpy library (for NPZ loading)
add_library(cnpy STATIC third_party/cnpy/cnpy.cpp)
target_link_libraries(cnpy ZLIB::ZLIB)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(SOURCES
    smpl_loader.cpp
    corrective_loader.cpp
    optimizer.cpp
    smpl_forward.cpp
)

# Executable
add_executable(smpl_pipeline
    main.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(smpl_pipeline
    cnpy
    Eigen3::Eigen
    Ceres::ceres
    gflags
    glog
)
