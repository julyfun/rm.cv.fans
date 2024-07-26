cmake_minimum_required(VERSION 3.15)

set(lib_name "base")

add_subdirectory("param")

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

aux_source_directory(./debug debug_src)
add_library(${lib_name} OBJECT ${debug_src})
target_link_libraries(${lib_name} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${lib_name})
