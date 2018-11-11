//
//  ptz_camera_util.h
//  EstimatePTZTripod
//
//  Created by jimmy on 11/9/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __EstimatePTZTripod__ptz_camera_util__
#define __EstimatePTZTripod__ptz_camera_util__

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "pgl_perspective_camera.h"
#include "gl_rotation_3d.h"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using std::vector;
using cvx_gl::rotation_3d;

namespace cvx_pgl {
    //@ brief estimate camera center and tripod rotation for a set of PTZ cameras.
    //  each camera provide 2D-3D correspondences,
    // init_common_rotation: rodrigues angle
    bool estimateCommomCameraCenterAndRotation(const vector<vector<Vector3d>> & wld_pts,
                                               const vector<vector<Vector2d>> & img_pts,
                                               const vector<perspective_camera> & init_cameras,
                                               const Vector3d & init_common_rotation,
                                               Vector3d & estimated_camera_center,
                                               Vector3d & estimated_common_rotation,
                                               vector<perspective_camera > & estimated_cameras);
    
    
    
    
}



#endif /* defined(__EstimatePTZTripod__ptz_camera_util__) */
