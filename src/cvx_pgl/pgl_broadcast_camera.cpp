//
//  pgl_broadcast_camera.cpp
//  EstimatePTZTripod
//
//  Created by jimmy on 11/14/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "pgl_broadcast_camera.h"


namespace cvx {
    broadcast_camera::broadcast_camera(int lambda_dim) {
        lambda_ = Eigen::VectorXd(lambda_dim);
        lambda_.setZero();
    }
    
    // @brief fl = 2000 is an arbitrary number
    broadcast_camera::broadcast_camera(const Vector2d& pp, const Vector3d& cc,
                                       const Vector3d& base_rot,
                                       const VectorXd& lambda,
                                       double pan, double tilt, double fl):ptz_camera(pp, cc, base_rot,
                                                                                       pan, tilt, fl) {
        assert(lambda.size() == 6 || lambda.size() == 12);
        lambda_ = lambda;
        recompute_matrix();
    }
    
    broadcast_camera::~broadcast_camera() {
        
    }
    
    void broadcast_camera::set_lambda(const VectorXd& lambda)
    {
        assert(lambda.size() == 6 || lambda.size() == 12);
        lambda_ = lambda;
        recompute_matrix();
    }
    
    Vector3d broadcast_camera::displacement(void) const
    {
        double fl = this->focal_length();
        double pan = this->pan();
        double tilt = this->tilt();
        Vector3d displacement;
        if (lambda_.size() == 6) {
            displacement = Vector3d(lambda_[0] + lambda_[3] * fl,
                                    lambda_[1] + lambda_[4] * fl,
                                    lambda_[2] + lambda_[5] * fl);
        }
        else if (lambda_.size() == 12) {
            displacement = Vector3d(lambda_[0] + lambda_[3] * fl + lambda_[6] * pan + lambda_[9] * tilt,
                                    lambda_[1] + lambda_[4] * fl + lambda_[7] * pan + lambda_[10] * tilt,
                                    lambda_[2] + lambda_[5] * fl + lambda_[8] * pan + lambda_[11] * tilt);
        }
        else {
            printf("Error: lambda must be 6 or 12\n");
        }
        
        return displacement;
    }    
    
    
    void broadcast_camera::recompute_matrix()
    {
        // Set new projection matrix to [ I | -C ]
        //     with 0, 0, 0, 1 as the last row
        Matrix4d Pnew;
        Pnew.setIdentity();
        Pnew(0,3) = -camera_center_.x();
        Pnew(1,3) = -camera_center_.y();
        Pnew(2,3) = -camera_center_.z();
        
        Matrix4d R;
        R.setIdentity();
        R.topLeftCorner(3, 3) = R_.as_matrix();
        
        Vector3d dis = this->displacement();
        Matrix34d dis_mat;
        dis_mat.setIdentity();
        dis_mat(0, 3) = dis.x();
        dis_mat(1, 3) = dis.y();
        dis_mat(2, 3) = dis.z();
        
        //printf("broadcast_camera::recompute_matrix \n");
        //  3 x 3, 3 x 4, 4 x 4, 4 x 4
        this->set_matrix(K_.get_matrix() * dis_mat * R * Pnew);
    }
    
}
