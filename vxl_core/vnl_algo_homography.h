//
//  vnl_algo_homography.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-05-29.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__vnl_algo_homography__
#define __CalibMeMatching__vnl_algo_homography__

#include <stdio.h>
#include "vnl_head_files.h"
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::Vector3d;
using Eigen::Vector2d;

class VnlAlgoHomography {
public:
    
    static
    bool estimateHomography(const vector<Vector2d> & src_pts,
                            const vector<Vector2d> & dst_pts,
                            const vector<Eigen::ParametrizedLine<double, 2> > & src_lines,
                            const vector<Vector2d> & dst_line_pts,
                            const Eigen::Matrix3d& init_homo,
                            Eigen::Matrix3d& final_homo);
    
};

#endif /* defined(__CalibMeMatching__vnl_algo_homography__) */
