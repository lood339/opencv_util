//
//  perspective_camera.h
//  calib
//
//  Created by jimmy on 2017-07-26.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __calib__vpgl_perspective_camera__
#define __calib__vpgl_perspective_camera__

#include <stdio.h>
#include <Eigen/Dense>
#include "pgl_proj_camera.h"
#include "pgl_calibration_matrix.h"

#include "gl_rotation_3d.h"


namespace cvx {
   
    class perspective_camera : public proj_camera
    {
        
    public:
        //: Default constructor
        // Makes a camera at the origin with no rotation and default calibration matrix.
        perspective_camera();
        ~perspective_camera();
        
        perspective_camera( const perspective_camera& cam );
        
        void set_calibration( const calibration_matrix& K );
        void set_camera_center( const Eigen::Vector3d& camera_center );
        void set_translation(const Eigen::Vector3d& t);
        void set_rotation(const Eigen::Vector3d& rodrigues);
        void set_rotation( const Eigen::Matrix3d& R );
        
        const calibration_matrix & get_calibration() const{ return K_; }
        const Eigen::Vector3d& get_camera_center() const { return camera_center_; }
        const rotation_3d& get_rotation() const{ return R_; }
        
        
        // new added
        Eigen::Vector2d project2d(const Eigen::Vector2d &p) const;
        Eigen::Vector2d project3d(const Eigen::Vector3d &p) const;
        
        
    protected:
        //: Recalculate the 3x4 camera matrix from the parameters.
        virtual void recompute_matrix();
        
        calibration_matrix K_;
        Eigen::Vector3d camera_center_;
        rotation_3d R_;
    };
    
} // namespace

#endif /* defined(__calib__vpgl_perspective_camera__) */
