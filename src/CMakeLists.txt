cmake_minimum_required(VERSION 3.6)

project(Final_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(final_project
    main.cpp
    ../Libs/cnpy/cnpy.cpp
)

target_include_directories(final_project PRIVATE
    ../Libs/cnpy
)

find_package(ZLIB REQUIRED)
target_link_libraries(final_project PRIVATE ZLIB::ZLIB)