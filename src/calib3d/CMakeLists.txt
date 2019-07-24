cmake_minimum_required (VERSION 3.5.1)

cmake_policy(SET CMP0015 NEW)

project (calib_tutorial)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11")
endif()

# The version number
set (calib_tutorial 1)
set (calib_tutorial 0)

set (USE_SERVER 0)
set (USE_ANACONDA 1)
set (USE_LINUX 0)
set (OPENCV_4 1)



# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

# conda path
if(USE_LINUX)
set (CONDA_DIR /home/jimmy/anaconda3/envs/pytorch_env)
else()
set (CONDA_DIR /Users/jimmy/anaconda3/envs/opencv_env)
endif()

# link OpenCV
set(OpenCV_DIR ${CONDA_DIR})

if (OPENCV_4)
include_directories ( ${OpenCV_DIR}/include/opencv4)
link_directories ( ${OpenCV_DIR}/lib )
else()
include_directories ( ${OpenCV_DIR}/include)
link_directories ( ${OpenCV_DIR}/lib )
endif()

# add directory files for head files?
#include_directories (.)

set(SOURCE_CODE calib_tutorial.cpp)

# add the executable
add_executable(tutorial  ${SOURCE_CODE})

# add library
target_link_libraries(tutorial opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d)










