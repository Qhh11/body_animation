cmake_minimum_required(VERSION 3.10)
project(SMPL_Project)

# Use C++17 for better math support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. Include Paths Setup
# ==========================================
# [FIX] Tell CMake to look in your local "include" folder
include_directories(include)

# Tell CMake to look in Libs/cnpy
include_directories("../Libs/cnpy")

# Tell CMake to look in Libs/Eigen (or system path)
include_directories("../Libs") 
include_directories("../Libs/Eigen") 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# ==========================================
# 2. Find Dependencies
# ==========================================
# A. Ceres Solver (Required)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# B. ZLIB (Required by cnpy)
find_package(ZLIB REQUIRED)

# ==========================================
# 3. Define Executable
# ==========================================
add_executable(main
    main.cpp
    smpl_model.cpp
    ../Libs/cnpy/cnpy.cpp 
)

# ==========================================
# 4. Link Libraries
# ==========================================
target_link_libraries(main
    ${CERES_LIBRARIES}  # Link Ceres
    ZLIB::ZLIB          # Link ZLIB
)

# Prevent Eigen alignment issues
add_definitions(-DEIGEN_NO_DEBUG)