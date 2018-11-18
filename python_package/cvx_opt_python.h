//
//  cvx_opt_python.h
//  EstimatePTZTripod
//
//  Created by jimmy on 11/11/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef EstimatePTZTripod_cvx_opt_python_h
#define EstimatePTZTripod_cvx_opt_python_h

// Interface for python

extern "C" {
    //@ estimate camera center and tripod rotation for PTZ cameras
    //model_pts: N * 3, 3d world coordinate
    //rows: N
    //cols: 3
    //input_init_cameras: M * 9, M is the number of cameras
    //camera_num: M
    // camera_param_len: 9
    // input_init_common_rotation: 3 x 1 , rodrigues angle
    // opt_cameras: M * 9, optimized perspective camera
    // opt_ptzs: M * 3, optimized pan, tilt and focal length
    // commom_center: camera center 3 * 1
    // commom_rotation: camera tripod rotation 3 * 1
    void estimateCommonCameraCenterAndRotation(const double* model_pts,
                                               const int rows,
                                               const int cols,
                                               const double* input_init_cameras,
                                               const int camera_num,
                                               const int camera_param_len,
                                               const double* input_init_common_rotation,
                                               double* opt_cameras,
                                               double* opt_ptzs,
                                               double* commom_center,
                                               double* commom_rotation);
    
    // @ brief estimate broadcasting camera parameters for a set of cameras.
    // camera model: PTZ camera + parameterized displacement
    // "Mimicking Human Camera Operators" WACV 2015
    // https://ieeexplore.ieee.org/document/7045890
    // model_3d_points: N * 3
    // point_num: N
    // init_cameras: M * 9, ppx, ppy, fl, rodx, rody, rodz, cx, cy, cz
    // output
    // opt_cameras: optimized camera parameters M * 9. This camera model is not accurate
    // opt_ptzs: optimized pan, tilt and focal length M * 3
    // shared_parameters: 12, camera center (3), common rotation (3), lambda (6)
    void estimateCommonCameraCenterAndRotationAndDisplacment(const double* model_3d_points,
                                                             const int point_num,
                                                             const double* init_cameras,
                                                             const int camera_num,
                                                             const double* init_common_rotation,
                                                             double* opt_cameras,
                                                             double* opt_ptzs,
                                                             double* shared_parameters);
    
    // camera_parameters: shared (camera center, rotation, lambda), principal point, pan-tilt-zoom,  12 + 2 + 3 = 17
    // model_3d_points: N * 3
    // point_num: N
    // output:
    // image_points: projected image location, N * 2
    void broadcastCameraProjection(const double* camera_parameters,
                                   const double* model_3d_points,
                                   const int point_num,
                                   double* image_points);
    
}

#endif
