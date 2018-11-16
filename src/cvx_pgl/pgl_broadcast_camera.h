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
//

namespace cvx {
    using Eigen::VectorXd;
    using Eigen::Vector3d;
    using Eigen::Matrix4d;
   
    class broadcast_camera : public ptz_camera {
         VectorXd lambda_;   // dimension is 6
        
    public:
        broadcast_camera();
        
        // @brief fl = 2000 is an arbitrary number
        broadcast_camera(const Vector2d& pp,
                         const Vector3d& cc,
                         const Vector3d& base_rot,
                         const VectorXd& lambda,
                         double pan = 0, double tilt = 0, double fl = 2000);
        ~broadcast_camera();
        
        // displacment between projection center and rotation center
        Vector3d displacement(void) const;

    protected:
        virtual void recompute_matrix();
        
         
        
        
    };
}


#endif /* defined(__EstimatePTZTripod__pgl_broadcast_camera__) */
