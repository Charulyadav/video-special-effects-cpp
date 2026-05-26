#Charul - 002535330
#Spring 2026
#CS 5330 Computer Vision
#Project 1: Video Special Effects

cmake_minimum_required(VERSION 3.16)
project(Project1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#OpenCV
set(OpenCV_DIR "C:/Users/Charul/Downloads/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#ONNX Runtime
set(ONNXRUNTIME_DIR "C:/Users/Charul/Downloads/onnxruntime-win-x64-1.26.0")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

#project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(vidDisplay src/vidDisplay.cpp src/filter.cpp src/faceDetect.cpp)
target_link_libraries(vidDisplay ${OpenCV_LIBS} onnxruntime)

add_executable(timeBlur src/timeBlur.cpp src/filter.cpp)
target_link_libraries(timeBlur ${OpenCV_LIBS} onnxruntime)