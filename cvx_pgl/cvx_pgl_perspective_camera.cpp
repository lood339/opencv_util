//
//  cvx_vpgl_perspective_camera.cpp
//  calib
//
//  Created by jimmy on 2017-07-26.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_pgl_perspective_camera.h"
#include <Eigen/Geometry>

namespace cvx_pgl {
    perspective_camera::perspective_camera()
    {
        
    }
    perspective_camera::~perspective_camera()
    {
        
    }
    
    perspective_camera::perspective_camera( const perspective_camera& other )
    {
        if (&other == this) {
            return;
        }
        K_ = other.K_;
        camera_center_ = other.camera_center_;
        R_ = other.R_;
    }
    
        
    void perspective_camera::set_calibration( const calibration_matrix& K )
    {
        K_ = calibration_matrix(K);
    }
    void perspective_camera::set_camera_center( const Eigen::Vector3d& camera_center )
    {
        camera_center_ = camera_center;        
    }
    void perspective_camera::set_translation(const Eigen::Vector3d& t)
    {
        Eigen::Matrix3d Rt = R_.as_matrix().transpose();
        Eigen::Vector3d cv = -(Rt * t);
        camera_center_ = cv;
    }
    
    void perspective_camera::set_rotation(const Eigen::Vector3d& rvector)
    {
        //@todo
        /*
        //printf("This function is not tested\n");
        double mag = rvector.norm();
        Eigen::Quaternion<double> q;
        if (mag > 0.0) {
            Eigen::Vector3d r = rvector/mag;
            Eigen::AngleAxisd aa(mag, r);
            q = Eigen::Quaternion<double>(aa);
        }
        else { // identity rotation is a special case
            q = Eigen::Quaternion<double>(1, 0, 0, 0);
        }
        R_ = q.normalized().toRotationMatrix();
         */
        R_ = rotation3d(rvector);
    }
    
    void perspective_camera::set_rotation( const Eigen::Matrix3d& R )
    {
        R_ = rotation3d(R);
    }
}