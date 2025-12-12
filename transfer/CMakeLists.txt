cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(transfer_mobilenet_v2 LANGUAGES CXX)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include(GNUInstallDirs)

add_executable (transfer transfer.cpp)
target_link_libraries(transfer ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
