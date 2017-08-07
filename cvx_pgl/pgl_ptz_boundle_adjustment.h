//
//  pgl_ptz_boundle_adjustment.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__pgl_ptz_boundle_adjustment__
#define __CalibMeMatching__pgl_ptz_boundle_adjustment__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include "pgl_ptz_camera.h"

// ptz boundle adjustment
using std::vector;
namespace cvx_pgl {
    class pgl_ptz_ba {
    public:
        pgl_ptz_ba()
        {
            max_iteration_ = 10000;
            reprojection_error_threshold_ = 2.0; // pixel
        }
        
        ~pgl_ptz_ba()
        {
            
        }
        
        void set_reprojection_error_criteria(double c) {reprojection_error_threshold_ = c;}
        
        // points: pan and tilt
        // visibility: point index
        // pp: principal point
        double run(vector<Eigen::Vector2d>& points,
                   const vector<vector<Eigen::Vector2d> >& image_points,
                   const vector<vector<int> >& visibility,
                   const Eigen::Vector2d & pp,
                   vector<Vector3d> & pan_tilt_fl);
        
    private:
        int max_iteration_;
        double reprojection_error_threshold_;
        
    };
    
    // loation1, location2: nx2, point correspondences in camera1 and camera2
    bool boundle_adjustment(ptz_camera& camera1, ptz_camera& camera2,
                            MatrixXd& loation1,  MatrixXd& location2);
    
}

#endif /* defined(__CalibMeMatching__pgl_ptz_boundle_adjustment__) */
