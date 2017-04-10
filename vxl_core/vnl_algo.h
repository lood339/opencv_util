//
//  vnl_algo.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-08.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__vnl_algo__
#define __PointLineReloc__vnl_algo__

#include <stdio.h>
#include "vnl_head_files.h"
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::Vector3d;

class VnlAlgo
{
public:
    // camera_pts: unit meter
    // estimate caemra pose from point to point correspondence
    // and point on line correspondence
    // camera pose: camera coordinate to world coordinate
    static
    bool estimateCameraPose(const vector<Vector3d> & camera_pts,
                            const vector<Vector3d> & world_pts,
                            const vector<Vector3d> & camera_line_start_pts,
                            const vector<Vector3d> & camera_line_end_pts,
                            const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,                            
                            const Eigen::Affine3d& init_camera,
                            Eigen::Affine3d& final_camera);
                            
    
};


#endif /* defined(__PointLineReloc__vnl_algo__) */
