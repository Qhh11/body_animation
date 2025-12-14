cmake_minimum_required(VERSION 3.10)
project(SMPL_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("../Libs/cnpy")

include_directories("../Libs") 
include_directories("../Libs/Eigen") 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


find_package(ZLIB REQUIRED)


add_executable(main
    main.cpp
    smpl_model.cpp
    ../Libs/cnpy/cnpy.cpp 
)

target_link_libraries(main
    ${CERES_LIBRARIES} 
    ZLIB::ZLIB          
)

add_definitions(-DEIGEN_NO_DEBUG)