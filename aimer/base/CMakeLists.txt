cmake_minimum_required(VERSION 3.15)

set(lib_name aimer_base)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})

# 搜集目录中所有源文件，存储到变量中
aux_source_directory(./math math_src)
aux_source_directory(./debug debug_src)
aux_source_directory(./robot robot_src)

# 编译变量的源文件并生成对象库
add_library(${lib_name} ${math_src} ${debug_src} ${robot_src})

# 链接库
target_link_libraries(${lib_name} ${OpenCV_LIBS} ${CERES_LIBRARIES})
