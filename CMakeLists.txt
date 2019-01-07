cmake_minimum_required (VERSION 2.6)

cmake_policy(SET CMP0015 NEW)

project (OpenCV_util_lib)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11")
endif()

# The version number
set (OpenCV_util_lib_MAJOR 1)
set (OpenCV_util_lib_MINOR 0)

set (USE_SERVER 0)
set (USE_ANACONDA 0)



# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")


# link Eigen
if(USE_SERVER)
set(Eigen_DIR /lib/eigen_3.2.6/include/eigen3)
include_directories (${Eigen_DIR})
else()
set(Eigen_DIR /Users/jimmy/Code/Eigen_3.3.5)
include_directories (${Eigen_DIR})
endif()

# link OpenCV
if(USE_ANACONDA)
set(OpenCV_DIR /Users/jimmy/anaconda3/envs/pytorch_env)
else()
set(OpenCV_DIR /Users/jimmy/Code/opencv-4.0.0/opencv_400)
endif()
include_directories ( ${OpenCV_DIR}/include/opencv4 )
link_directories ( ${OpenCV_DIR}/lib )


# add directory files for head files?
include_directories (./src ./src/cvx_gl ./src/cvx_pgl ./src/cvx_opt
                     ./src/third_part/walsh_hadamard)


# .cpp in cvx_gl
set(SOURCE_CVX_GL 
	./src/cvx_gl/gl_homg_point_2d.cpp 
	./src/cvx_gl/gl_homg_point_3d.cpp 
	./src/cvx_gl/gl_ray_3d.cpp)

# .cpp in cvx_pgl
set(SOURCE_CVX_PGL 
	./src/cvx_pgl/pgl_calibration_matrix.cpp 
	./src/cvx_pgl/pgl_perspective_camera.cpp 
	./src/cvx_pgl/pgl_proj_camera.cpp
	./src/cvx_pgl/pgl_ptz_camera.cpp
	./src/cvx_pgl/pgl_broadcast_camera.cpp)

# .cpp in cvx_opt
set(SOURCE_CVX_OPT
	./src/cvx_opt/opt_homography.cpp 
	./src/cvx_opt/opt_perspective_camera.cpp 
	./src/cvx_opt/opt_ptz_camera.cpp
	./src/cvx_opt/opt_broadcast_camera.cpp)

# .cpp in Walsh Hadamard Transform
set(SOURCE_WHT ./src/third_part/walsh_hadamard/defs.c
    ./src/third_part/walsh_hadamard/image.c
    ./src/third_part/walsh_hadamard/matrix.c
    ./src/third_part/walsh_hadamard/wh.c
    ./src/cvx_walsh_hadamard.cpp)


#set(SOURCE_CODE ${SOURCE_CODE1} ${SOURCE_CODE_DT_UTIL})

# add the executable
#add_executable(DTC_train ${TRAIN_SRC} ${SOURCE_CODE})
#add_executable(DTC_test ${TEST_SRC} ${SOURCE_CODE})

# add library
add_library(cvx SHARED ${SOURCE_CVX_GL} ${SOURCE_CVX_PGL} ${SOURCE_CVX_OPT}
            ${SOURCE_WHT})
target_link_libraries(cvx opencv_core opencv_imgproc opencv_imgcodecs)


# for python interface
include_directories (./python_package)
set(SOURCE_CVX_OPT_PYTHON ./python_package/cvx_opt_python.cpp)
add_library(cvx_opt_python SHARED ${SOURCE_CVX_GL} 
		${SOURCE_CVX_PGL} 
		${SOURCE_CVX_OPT}
		${SOURCE_CVX_OPT_PYTHON} )


set(SOURCE_CVX_WHT_PYTHON ./python_package/cvx_wht_python.cpp)
add_library(cvx_wht_python SHARED ${SOURCE_WHT} ${SOURCE_CVX_WHT_PYTHON})
target_link_libraries(cvx_wht_python opencv_core opencv_imgproc)


















