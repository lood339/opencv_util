import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
#@todo hardcode library
lib = cdll.LoadLibrary('/Users/jimmy/Code/opencv_util/build/libcvx_opt_python.dylib')

def optimize_ptz_cameras(model_3d_points, init_cameras, init_rod):
    """
    optimize pan-tilt-zoom camera parameters using LMQ mehtod
    :param model_3d_points: N * 3
    :param init_cameras:  M * 9
    :param init_rod:  3 * 1
    :return: N * 9 cameras, N * 3 pan-tilt-zooms,
            3 * 1 camera_center,
            3 * 1 base rotation
    """
    assert model_3d_points.shape[1] == 3
    assert init_cameras.shape[1] == 9
    assert init_rod.shape[0] == 3

    point_num = model_3d_points.shape[0]
    cols = 3
    camera_num = init_cameras.shape[0]
    camera_param_len = 9

    opt_cameras = np.zeros((camera_num, 9))
    opt_ptzs = np.zeros((camera_num, 3))
    common_center = np.zeros((3, 1))
    common_rotation = np.zeros((3, 1))
    lib.estimateCommonCameraCenterAndRotation(c_void_p(model_3d_points.ctypes.data),
                                              c_int(point_num),
                                              c_int(cols),
                                              c_void_p(init_cameras.ctypes.data),
                                              c_int(camera_num),
                                              c_int(camera_param_len),
                                              c_void_p(init_rod.ctypes.data),
                                              c_void_p(opt_cameras.ctypes.data),
                                              c_void_p(opt_ptzs.ctypes.data),
                                              c_void_p(common_center.ctypes.data),
                                              c_void_p(common_rotation.ctypes.data))
    return (opt_cameras, opt_ptzs, common_center, common_rotation)


def optimize_broadcast_cameras(model_3d_points, init_cameras, init_rod, lambda_dim):
    """
    optimize broadcast camera parameters using LMQ mehtod
    It is slow
    :param model_3d_points: N * 3
    :param init_cameras:  M * 9
    :param init_rod:  3 * 1
    :param lamba_dim: 6 or 12. linear weight of pan-tilt-zoom with displacement
    :return: N * 9 cameras, N * 3 pan-tilt-zooms,
        12 * 1 or 18 * 1 shared parameters, camera_center, base rotation and lambda
    """
    assert model_3d_points.shape[1] == 3
    assert init_cameras.shape[1] == 9
    assert init_rod.shape[0] == 3
    assert lambda_dim == 6 or lambda_dim == 12

    point_num = model_3d_points.shape[0]
    camera_num = init_cameras.shape[0]

    opt_cameras = np.zeros((camera_num, 9))
    opt_ptzs = np.zeros((camera_num, 3))
    shared_parameters = np.zeros((6 + lambda_dim, 1))
    lib.estimateCommonCameraCenterAndRotationAndDisplacment(c_void_p(model_3d_points.ctypes.data),
                                                            c_int(point_num),
                                                            c_void_p(init_cameras.ctypes.data),
                                                            c_int(camera_num),
                                                            c_void_p(init_rod.ctypes.data),
                                                            c_int(lambda_dim),
                                                            c_void_p(opt_cameras.ctypes.data),
                                                            c_void_p(opt_ptzs.ctypes.data),
                                                            c_void_p(shared_parameters.ctypes.data))
    return (opt_cameras, opt_ptzs, shared_parameters)

def broadcast_camera_projection(common_param, pp, ptz, points):
    """
    :param common_param: 12 or 18
        shared (camera center, rotation, lambda), 3 + 3 + 6 or 12
    pp: principal point, 
    ptz: 3
    :param points: N * 3, 3d point
    :return: 2d image locations
    """
    assert common_param.shape[0] == 12 or common_param.shape[0] == 18
    assert pp.shape[0] == 2
    assert ptz.shape[0] == 3
    assert points.shape[1] == 3

    camera = np.vstack((common_param, pp, ptz))
    N = points.shape[0]
    image_points = np.zeros((N, 2))
    lambda_dim = common_param.shape[0] - 6

    lib.broadcastCameraProjection(c_void_p(camera.ctypes.data),
                                  c_int(lambda_dim),
                                  c_void_p(points.ctypes.data),
                                  c_int(N),
                                  c_void_p(image_points.ctypes.data))
    return image_points
