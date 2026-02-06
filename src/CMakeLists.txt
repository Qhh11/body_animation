cmake_minimum_required(VERSION 3.10)
project(SMPL_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Paths ----
# If this CMakeLists.txt sits in .../Exercises/src, then:
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# 1. Include Paths

include_directories(${PROJ_ROOT}/include)
include_directories(${PROJ_ROOT}/../Libs/cnpy)
include_directories(${PROJ_ROOT}/../Libs/Eigen)
include_directories(${PROJ_ROOT}/../Libs)

# 2. Find Packages
find_package(Eigen3 REQUIRED)
# Prefer imported target if available; still keep include for compatibility
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(ZLIB REQUIRED)

# 3. Executable
add_executable(main
    ${PROJ_ROOT}/main.cpp
    ${PROJ_ROOT}/smpl_model.cpp
    ${PROJ_ROOT}/optimizer.cpp
    ${PROJ_ROOT}/../Libs/cnpy/cnpy.cpp
)

# Extra sources (NOTE: NO "src/" prefix)
target_sources(main PRIVATE
  ${PROJ_ROOT}/io_utils.cpp
  ${PROJ_ROOT}/openpose_io.cpp
  ${PROJ_ROOT}/init_utils.cpp
  ${PROJ_ROOT}/config_builders.cpp
  ${PROJ_ROOT}/threedpw_gt.cpp
  ${PROJ_ROOT}/amass_runners.cpp
  ${PROJ_ROOT}/openpose_runners.cpp
)

# 4. Linking
target_link_libraries(main
    ${CERES_LIBRARIES}
    ZLIB::ZLIB
)

target_include_directories(main PRIVATE
  ${PROJ_ROOT}/include
)

add_definitions(-DEIGEN_NO_DEBUG)
