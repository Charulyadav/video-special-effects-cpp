#Charul - 002535330
#Spring 2026
#CS 5330 Computer Vision
#Project 1: Video Special Effects

cmake_minimum_required(VERSION 3.16)
project(Project1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

find_package(OpenCV REQUIRED)

set(PROJECT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

add_library(filters STATIC src/filter.cpp)
target_include_directories(filters PUBLIC "${PROJECT_INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(filters PUBLIC ${OpenCV_LIBS})

add_executable(timeBlur src/timeBlur.cpp)
target_link_libraries(timeBlur PRIVATE filters)

option(BUILD_ONNX_EFFECTS "Build ONNX Runtime based video effects" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime installation")

if(BUILD_ONNX_EFFECTS)
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to your ONNX Runtime install, or configure with -DBUILD_ONNX_EFFECTS=OFF")
    endif()

    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include"
        NO_DEFAULT_PATH)

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH)

    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "Could not find ONNX Runtime under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
    endif()

    add_executable(vidDisplay src/vidDisplay.cpp src/faceDetect.cpp)
    target_include_directories(vidDisplay PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(vidDisplay PRIVATE filters "${ONNXRUNTIME_LIBRARY}")
endif()
