cmake_minimum_required(VERSION 3.15)

set(lib_name auto_aim_predictor)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})

# 搜集目录中所有源文件，存储到变量中
aux_source_directory(./aim aim_src)
aux_source_directory(./enemy enemy_src)
aux_source_directory(./motion motion_src)
aux_source_directory(./pnp pnp_src)
aux_source_directory(./enemy_predictor enemy_predictor_src)
aux_source_directory(. src)

# 编译变量的源文件并生成对象库 该库完全描述一个线程，不需要被链接
add_library(${lib_name} OBJECT ${aim_src} ${enemy_src} ${motion_src} ${pnp_src}
    ${enemy_predictor_src} ${src})

# 链接库
target_link_libraries(${lib_name} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(${lib_name} aimer_base)
target_link_libraries(${lib_name} base_param)

target_link_libraries(${PROJECT_NAME} ${lib_name})
