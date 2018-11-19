import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
#@todo hardcode library
lib = cdll.LoadLibrary('/Users/jimmy/Source/opencv_util/build/libcvx_opt_python.dylib')


def optimize_broadcast_cameras(model_3d_points, init_cameras, init_rod):
    """
    optimize broadcast camera parameters using LMQ mehtod
    It is slow
    :param model_3d_points: N * 3
    :param init_cameras:  M * 9
    :param init_rod:  3 * 1
    :return: N * 9 cameras, N * 3 pan-tilt-zooms,
        12 * 1 shared parameters, camera_center, base rotation and lambda
    """
    assert model_3d_points.shape[1] == 3
    assert init_cameras.shape[1] == 9
    assert init_rod.shape[0] == 3

    point_num = model_3d_points.shape[0]
    camera_num = init_cameras.shape[0]

    opt_cameras = np.zeros((camera_num, 9))
    opt_ptzs = np.zeros((camera_num, 3))
    shared_parameters = np.zeros((12, 1))
    lib.estimateCommonCameraCenterAndRotationAndDisplacment(c_void_p(model_3d_points.ctypes.data),
                                                            c_int(point_num),
                                                            c_void_p(init_cameras.ctypes.data),
                                                            c_int(camera_num),
                                                            c_void_p(init_rod.ctypes.data),
                                                            c_void_p(opt_cameras.ctypes.data),
                                                            c_void_p(opt_ptzs.ctypes.data),
                                                            c_void_p(shared_parameters.ctypes.data))
    return (opt_cameras, opt_ptzs, shared_parameters)

def broadcast_camera_projection(camera, points):
    """
    :param camera: 17 parameters
        shared (camera center, rotation, lambda), principal point, pan-tilt-zoom,  12 + 2 + 3 = 17
    :param points: N * 3, 3d point
    :return: 2d image locations
    """
    assert camera.shape[0] == 17
    assert points.shape[1] == 3

    N = points.shape[0]
    image_points = np.zeros((N, 2))
    lib.broadcastCameraProjection(c_void_p(camera.ctypes.data),
                                  c_void_p(points.ctypes.data),
                                  c_int(N),
                                  c_void_p(image_points.ctypes.data))
    return image_points