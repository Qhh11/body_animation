cmake_minimum_required(VERSION 3.10)
project(SMPL_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Include Paths
include_directories(include)
include_directories("../Libs/cnpy")
include_directories("../Libs/Eigen") 
include_directories("../Libs") 

# 2. Find Packages
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(ZLIB REQUIRED)

# 3. Executable
# Removed render.cpp, added optimizer.cpp
add_executable(main
    main.cpp
    smpl_model.cpp
    optimizer.cpp
    ../Libs/cnpy/cnpy.cpp 
)

# 4. Linking
target_link_libraries(main
    ${CERES_LIBRARIES}
    ZLIB::ZLIB
)

add_definitions(-DEIGEN_NO_DEBUG)