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
#include "cvx_gl_rotation_3d.h"
#include "cvx_pgl_calibration_matrix.h"

namespace cvx_pgl {
   
    class perspective_camera
    {
        using rotation3d = cvx_gl::rotation_3d;
        using calibration_matrix = cvx_pgl::calibration_matrix;
        
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
       // vgl_vector_3d<T> get_translation() const;
        const rotation3d& get_rotation() const{ return R_; }
        
        //double focal_length() const { return K_(0, 0); }
        //Eigen::Vector2d principal_point() const { return Eigen::Vector2d(K_(0, 2), K_(1, 2)); }
        
        
        
        
        /* @todo
         template <class T>
         void vpgl_perspective_camera<T>::recompute_matrix()
         {
         vnl_matrix_fixed<T,3,4> Pnew( (T)0 );
         
         // Set new projection matrix to [ I | -C ].
         for ( int i = 0; i < 3; i++ )
         Pnew(i,i) = (T)1;
         Pnew(0,3) = -camera_center_.x();
         Pnew(1,3) = -camera_center_.y();
         Pnew(2,3) = -camera_center_.z();
         
         // Then multiply on left to get KR[ I | -C ].
         this->set_matrix(K_.get_matrix()*R_.as_matrix()*Pnew);
         }

         */
    private:
        calibration_matrix K_;
        Eigen::Vector3d camera_center_;
        rotation3d R_;
    };
    
} // namespace

#endif /* defined(__calib__vpgl_perspective_camera__) */
