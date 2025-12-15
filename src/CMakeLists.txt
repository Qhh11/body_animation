cmake_minimum_required(VERSION 3.10)
project(SMPL_Project)

# 使用 C++17 以获得更好的数学库支持
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. 设置头文件路径 (Include Paths)
# ==========================================
# 1.1 项目本地的头文件 (cost_functions.h, optimizer.h 等)
include_directories(include)

# 1.2 [关键修改] 让编译器去上级目录的 Libs/cnpy 里找 cnpy.h
include_directories(../Libs/cnpy)

# 1.3 通用 Libs 路径
include_directories(../Libs) 
include_directories("../Libs/Eigen") 

# 1.4 Eigen (如果系统已安装则优先使用系统路径)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# ==========================================
# 2. 寻找依赖库 (Dependencies)
# ==========================================
# Ceres Solver (优化核心)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# ZLIB (cnpy 需要它)
find_package(ZLIB REQUIRED)

# ==========================================
# 3. 定义可执行文件 (Sources)
# ==========================================
# 注意：这里我们移除了 render.cpp，加入了 optimizer.cpp
# 并且直接编译 ../Libs/cnpy/cnpy.cpp
add_executable(main
    main.cpp
    smpl_model.cpp
    optimizer.cpp
    ../Libs/cnpy/cnpy.cpp 
)

# ==========================================
# 4. 链接库文件 (Linking)
# ==========================================
target_link_libraries(main
    ${CERES_LIBRARIES}  # 链接 Ceres
    ZLIB::ZLIB          # 链接 ZLIB (给 cnpy 用)
)

# 防止 Eigen 的一些对齐报错
add_definitions(-DEIGEN_NO_DEBUG)