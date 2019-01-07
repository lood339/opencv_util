//
//  pgl_broadcast_camera.h
//  EstimatePTZTripod
//
//  Created by jimmy on 11/14/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __EstimatePTZTripod__pgl_broadcast_camera__
#define __EstimatePTZTripod__pgl_broadcast_camera__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include "pgl_ptz_camera.h"

// broadcast camera: the rotation center is different from the
// projection center.
// dx = \lambda_1 + \lambda_4 * f
// dy = \lambda_2 + \lambda_5 * f
// dz = \lambda_3 + \lambda_6 * f
// or
// dx = \lambda_1 + \lambda_4 * f + \lambda_7 * pan + \lambda_10 * tilt
// dy = \lambda_2 + \lambda_5 * f + \lambda_8 * pan + \lambda_11 * tilt
// dz = \lambda_3 + \lambda_6 * f + \lambda_9 * pan + \lambda_12 * tilt

namespace cvx {
    using Eigen::VectorXd;
    using Eigen::Vector3d;
    using Eigen::Matrix4d;
   
    class broadcast_camera : public ptz_camera {
         VectorXd lambda_;   // dimension is 6 or 12
        
    public:
        broadcast_camera(int lambda_dim = 6);
        
        // @brief fl = 2000 is an arbitrary number
        broadcast_camera(const Vector2d& pp,
                         const Vector3d& cc,
                         const Vector3d& base_rot,
                         const VectorXd& lambda,
                         double pan = 0, double tilt = 0, double fl = 2000);
        ~broadcast_camera();
        
        void set_lambda(const VectorXd& lambda);
        VectorXd lambda(void) const {return lambda_;}
        
        // displacment between projection center and rotation center
        Vector3d displacement(void) const;

    protected:
        virtual void recompute_matrix();        
    };
}


#endif /* defined(__EstimatePTZTripod__pgl_broadcast_camera__) */
