
import scipy.io as sio
import numpy as np
import ctypes
from ctypes import cdll
from ctypes import c_int
lib = cdll.LoadLibrary('../build/libcvx_opt_python.dylib')


# prepare data
model = sio.loadmat('./data/ice_hockey_edge_point_6_feet.mat')
edge_points = model['edge_points']
model_pts = np.zeros((edge_points.shape[0], 3))
model_pts[:, 0:2] = edge_points
model_pts[:,2] = 0.0
rows, cols = model_pts.shape


import glob
files = glob.glob('./data/annotation/*.txt')
N = len(files)
# initial camera data

init_cameras = np.zeros((N, 9))
for i in range(N):
    file_name = files[i]
    data = np.loadtxt(file_name, delimiter='\t', skiprows=2)
    init_cameras[i, :] = data

#sio.savemat('init_cameras.mat', {'init_camera':init_cameras})


camera_num, camera_param_len = N, 9

init_commont_rotation = np.zeros((3, 3))

# main camera in the center
init_commont_rotation[0][0] = 1.0
init_commont_rotation[1][2] = -1.0
init_commont_rotation[2][1] = 1.0

import cv2 as cv
rod = np.zeros((3, 1))
cv.Rodrigues(init_commont_rotation, rod)


opt_cameras = np.zeros((N, 9))
common_center = np.zeros((3, 1))
common_rotation = np.zeros((3, 1))


lib.estimateCommomCameraCenterAndRotation(ctypes.c_void_p(model_pts.ctypes.data),
                                          c_int(rows),
                                          c_int(cols),
                                          ctypes.c_void_p(init_cameras.ctypes.data),
                                          c_int(camera_num),
                                          c_int(camera_param_len),
                                          ctypes.c_void_p(rod.ctypes.data),
                                          ctypes.c_void_p(opt_cameras.ctypes.data),
                                          ctypes.c_void_p(common_center.ctypes.data),
                                          ctypes.c_void_p(common_rotation.ctypes.data))

# visualize result







