//
//  opt_broadcast_camera.h
//  EstimatePTZTripod
//
//  Created by jimmy on 11/15/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __EstimatePTZTripod__opt_broadcast_camera__
#define __EstimatePTZTripod__opt_broadcast_camera__

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "pgl_broadcast_camera.h"
#include "gl_rotation_3d.h"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using std::vector;
using cvx::rotation_3d;
using cvx::perspective_camera;
using cvx::broadcast_camera;

namespace cvx {
    //@ brief estimate camera center and tripod rotation, and displacement function for a set of PTZ cameras.
    //  each camera provide 2D-3D correspondences,
    // init_common_rotation: rodrigues angle
    bool estimateBroadcastCameras(const vector<vector<Vector3d>> & wld_pts,
                                  const vector<vector<Vector2d>> & img_pts,
                                  const vector<perspective_camera> & init_cameras,
                                  const Vector3d & init_common_rotation,
                                  vector<broadcast_camera> & estimated_cameras);
                            
    
}

#endif /* defined(__EstimatePTZTripod__opt_broadcast_camera__) */
