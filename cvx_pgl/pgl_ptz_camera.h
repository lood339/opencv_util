//
//  pgl_ptz_camera.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__pgl_ptz_camera__
#define __CalibMeMatching__pgl_ptz_camera__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include "pgl_perspective_camera.h"

// camera model from "Mimicking Human Camera Operators" from WACV 2015
namespace cvx_pgl {
    using Eigen::Vector2d;
    using Eigen::Vector3d;
    using std::vector;
    using Eigen::MatrixXd;
    class ptz_camera :public perspective_camera {
        Vector2d     pp_;     // principle point
        Vector3d     cc_;     // camera center
        Vector3d     base_rotation_;     // camera base rotation, rodrigues angle
        
        Vector3d     ptz_; // pan, tilt and focal length, angles in degree
    public:
        ptz_camera(){}
        
        ptz_camera(const Vector2d& pp, const Vector3d& cc,
                   const Vector3d& base_rot, double pan, double tilt, double fl):pp_(pp),
        cc_(cc), base_rotation_(base_rot), ptz_(pan, tilt, fl){}
        
        // assume common parameters are fixed.
        // convert general perspective camera to ptz camera
        // wld_pts: nx3 matrix, world coordinate
        // img_pts: nx2 matrix, image coordinate
        bool set_camera(const perspective_camera& camera,
                        const MatrixXd & wld_pts,
                        const MatrixXd & img_pts);
        
        
        
        static bool estimatePTZWithFixedBasePositionRotation (const MatrixXd & wld_pts,
                                                              const MatrixXd & img_pts,
                                                              const perspective_camera & init_camera,
                                                              const Vector3d & camera_center,
                                                              const Vector3d & rod,
                                                              Vector3d & ptz,
                                                              perspective_camera & estimated_camera);

    };
    
    // pan, tilt: degree
    Eigen::Matrix3d matrixFromPanYTiltX(double pan, double tilt);
}


#endif /* defined(__CalibMeMatching__pgl_ptz_camera__) */
