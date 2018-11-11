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


# add directory files for head files?
include_directories (./src ./src/cvx_gl ./src/cvx_pgl ./src/cvx_opt)


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
	./src/cvx_pgl/pgl_ptz_camera.cpp)

# .cpp in cvx_opt
set(SOURCE_CVX_OPT
	./src/cvx_opt/opt_homography.cpp 
	./src/cvx_opt/opt_perspective_camera.cpp 
	./src/cvx_opt/opt_ptz_camera.cpp)



# .cpp in dt_util
#set(SOURCE_CODE_DT_UTIL ./src/dt_util/dt_random.cpp ./src/dt_util/dt_param_parser ./src/dt_util/dt_util.cpp ./src/dt_util/vnl_random.cxx)

#set(SOURCE_CODE ${SOURCE_CODE1} ${SOURCE_CODE_DT_UTIL})

# add the executable
#add_executable(DTC_train ${TRAIN_SRC} ${SOURCE_CODE})
#add_executable(DTC_test ${TEST_SRC} ${SOURCE_CODE})

# add library
add_library(cvx SHARED ${SOURCE_CVX_GL} ${SOURCE_CVX_PGL} ${SOURCE_CVX_OPT})
















