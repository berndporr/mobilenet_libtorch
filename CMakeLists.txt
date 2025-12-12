cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(mobilenet_v2 LANGUAGES CXX)
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

add_subdirectory(transfer)

add_executable (demo_mobilenet demo_mobilenet.cpp)
target_link_libraries(demo_mobilenet ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(torch-mobilenetv2 INTERFACE)
target_include_directories(torch-mobilenetv2 INTERFACE .)

set_property(TARGET torch-mobilenetv2
  PROPERTY PUBLIC_HEADER mobilenet_v2.h)

install(TARGETS torch-mobilenetv2 PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
