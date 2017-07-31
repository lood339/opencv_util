//
//  cvx_gl_homography.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-07-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__cvx_gl_homography__
#define __CalibMeMatching__cvx_gl_homography__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using Eigen::Vector2d;
using std::vector;

namespace cvx_gl {
    // @brief refine homography using point-to-point and point-on-line correspondences
    bool estimateHomography(const vector<Vector2d> & src_pts,
                            const vector<Vector2d> & dst_pts,
                            const vector<Eigen::ParametrizedLine<double, 2> > & src_lines,
                            const vector<Vector2d> & dst_line_pts,
                            const Eigen::Matrix3d& init_homo,
                            Eigen::Matrix3d& final_homo);
    
    
    
}

#endif /* defined(__CalibMeMatching__cvx_gl_homography__) */
