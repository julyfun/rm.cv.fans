cmake_minimum_required(VERSION 3.15)

# 构建子目录
if(${network_engine} STREQUAL "opencv")
    add_subdirectory("detector_opencv")
elseif(${network_engine} STREQUAL "cuda")
    add_subdirectory("detector_cuda")
elseif(${network_engine} STREQUAL "openvino")
endif()

add_subdirectory("predictor")
