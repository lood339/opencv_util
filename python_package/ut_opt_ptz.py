from sklearn.datasets import make_classification
import scipy.io as sio
import numpy as np

import ctypes
from ctypes import cdll
from ctypes import c_int
lib = cdll.LoadLibrary('./build/libcvx_opt_python.dylib')


#param_name = c_char_p("/Users/jimmy/Desktop/learn_program/mt_dtc/dtc_tree_param.txt".encode('utf-8'))
#save_name = c_char_p("debug.txt".encode('utf-8'))

# prepare data
model = sio.loadmat('./data/ice_hockey_edge_point_6_feet.mat')
edge_points = model['edge_points']
print(edge_points.shape)

model_pts = np.zeros((edge_points.shape[0], 3))
model_pts[:, 0:2] = edge_points
model_pts[:,2] = 0.0

rows, cols = model_pts.shape

#print(model_pts)

# initial camera data
frames = [1, 25, 100, 125, 150]
N = len(frames)
init_cameras = np.zeros((len(frames), 9))
for i in range(len(frames)):
    file_name = './data/{}_manual_calib.txt'.format(frames[i])
    data = np.loadtxt(file_name, delimiter='\t', skiprows=2)
    init_cameras[i, :] = data
print(init_cameras.shape)

#sio.savemat('init_cameras.mat', {'init_camera':init_cameras})


camera_num, camera_param_len = len(frames), 9

init_commont_rotation = np.zeros((3, 3))

# main camera in the center
init_commont_rotation[0][0] = 1.0
init_commont_rotation[1][2] = -1.0
init_commont_rotation[2][1] = 1.0

import cv2 as cv
rod = np.zeros((3, 1))
cv.Rodrigues(init_commont_rotation, rod)
print(rod)

opt_cameras = np.zeros((N, 3))
common_center = np.zeros((3, 1))
common_rotation = np.zeros((3, 1))

"""
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


"""
"""
lib.dtc_train(ctypes.c_void_p(X_train.ctypes.data),
              ctypes.c_void_p(Y_train.ctypes.data),
              X_train.shape[0],
              X_train.shape[1],
              param_name,
              save_name)
"""



