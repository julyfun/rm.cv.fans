cmake_minimum_required(VERSION 3.15)

set(LIB_NAME armor_detector)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

# find_package(fmt REQUIRED)
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

aux_source_directory(. src)

# add_executable(yolov5_trt main.cpp
# ../infantry2021-0506/armor/detector/TRTModule.cpp MindVision/MindVision.cpp
# ${TensorRT_SOURCE})
add_library(${LIB_NAME} OBJECT entry.cpp ${src} ${TensorRT_SOURCE})

# target_link_libraries(${LIB_NAME} ${TensorRT_LIBS} ${OpenCV_LIBS} fmt::fmt
# nvonnxparser MVSDK pthread cuda)
# [WARNING] @florian: 这里 MVSDK 可能是仅仅适配 hikvision 的
target_link_libraries(${LIB_NAME} ${TensorRT_LIBS} ${OpenCV_LIBS}
    nvonnxparser
    pthread
    cuda)

target_link_libraries(${PROJECT_NAME} ${LIB_NAME})
