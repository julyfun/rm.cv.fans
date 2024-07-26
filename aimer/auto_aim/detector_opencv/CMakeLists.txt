cmake_minimum_required(VERSION 3.15)

set(LIB_NAME auto_aim_detector)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

aux_source_directory(. src)

add_library(${LIB_NAME} OBJECT entry.cpp ${src})

# target_link_libraries(${LIB_NAME} ${TensorRT_LIBS} ${OpenCV_LIBS} fmt::fmt
# nvonnxparser MVSDK pthread cuda)
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS} pthread)

target_link_libraries(${PROJECT_NAME} ${LIB_NAME})
