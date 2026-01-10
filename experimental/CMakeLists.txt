cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(transfer_mobilenet_v2 LANGUAGES CXX)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Libtorch version: ${Torch_VERSION}")
message(STATUS "Libtorch include path: ${TORCH_INCLUDE_DIRS}")


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include(GNUInstallDirs)

add_executable (transfer transfer.cpp)
target_link_libraries(transfer ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES}
    extension_tensor
    extension_module
    extension_data_loader
    extension_flat_tensor
    executorch
    executorch_core

    # XNNPACK backend
    # xnnpack_backend
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>"

    # XNNPACK dependencies
    XNNPACK

    xnnpack-microkernels-prod

    pthreadpool
    extension_threadpool

    quantized_kernels
    quantized_ops_lib
    kernels_util_all_deps
    cpuinfo
)
