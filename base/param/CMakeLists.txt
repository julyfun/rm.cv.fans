cmake_minimum_required(VERSION 3.15)

set(lib_name base_param)
set(entry_name base_param_entry)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 同一个 cmake 下共享??
add_library(${lib_name} parameter.cpp)
add_library(${entry_name} OBJECT entry.cpp)

target_link_libraries(${lib_name} ${OpenCV_LIBS})
target_link_libraries(${entry_name} ${lib_name})
target_link_libraries(${PROJECT_NAME} ${entry_name})
