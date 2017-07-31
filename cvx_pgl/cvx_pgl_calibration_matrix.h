//
//  cvx_pgl_calibration_matrix.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-07-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__cvx_pgl_calibration_matrix__
#define __CalibMeMatching__cvx_pgl_calibration_matrix__

#include <stdio.h>
#include <Eigen/Dense>

using Eigen::Vector2d;
using Eigen::Matrix3d;

namespace cvx_pgl {
    class calibration_matrix {
        
    public:
        //: Default constructor makes an identity matrix.
        calibration_matrix();
        
        //: Destructor
        virtual ~calibration_matrix() {}
        
        calibration_matrix(double focal_length,
                           const Eigen::Vector2d& principal_point, double x_scale, double y_scale, double skew );
        
        calibration_matrix( const Eigen::Matrix3d& K );

        
    protected:
        //: The following is a list of the parameters in the calibration matrix.
        double focal_length_;
        Vector2d principal_point_;
        double x_scale_, y_scale_, skew_;
    };
    
} // namespace

#endif /* defined(__CalibMeMatching__cvx_pgl_calibration_matrix__) */
