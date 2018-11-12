cmake_minimum_required (VERSION 2.6)

cmake_policy(SET CMP0015 NEW)

project (cvx_opt_python_lib)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11")
endif()

# The version number
set (cvx_opt_python_lib_MAJOR 1)
set (cvx_opt_python_lib_MINOR 0)

set (USE_SERVER 0)


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
link_directories(${CMAKE_SOURCE_DIR})


# link Eigen
if(USE_SERVER)
set(Eigen_DIR /lib/eigen_3.2.6/include/eigen3)
include_directories (${Eigen_DIR})
else()
set(Eigen_DIR /Users/jimmy/Code/Eigen_3.3.5)
include_directories (${Eigen_DIR})
endif()

set(OPENCV_UTIL_SRC /Users/jimmy/Source/opencv_util/src)

# add directory files for head files?
include_directories (${OPENCV_UTIL_SRC}/cvx_gl
                     ${OPENCV_UTIL_SRC}/cvx_pgl
                     ${OPENCV_UTIL_SRC}/cvx_opt)


# .cpp
set(SOURCE_CODE ./cvx_opt_python.cpp)


# add the executable
#add_executable(DTC_train ${TRAIN_SRC} ${SOURCE_CODE})
#add_executable(DTC_test ${TEST_SRC} ${SOURCE_CODE})

# add library
add_library(cvx_opt_python SHARED ${SOURCE_CODE})
target_link_libraries(cvx_opt_python cvx)

















