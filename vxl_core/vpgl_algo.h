//
//  vpgl_algo.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-11.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__vpgl_algo__
#define __PointLineReloc__vpgl_algo__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::Vector3d;
using Eigen::Vector2d;

class VpglAlgo
{
    
public:
    // non-linear optimization for camera pose
    // init_pose: 3x4 camera pose from world to camera coordinate
    // init_pose has to close to the final (ground truth) pose
    static
    bool estimateCameraPose(const vector<Vector2d> & img_pts,
                            const vector<Vector3d> & world_pts,
                            
                            const vector<vector<Vector2d > > & image_line_pts,
                            const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,
                            
                            const Eigen::Matrix3d& camera_matrix,
                            const Eigen::Affine3d& init_pose,
                            Eigen::Affine3d& refined_pose);
    
};


Eigen::ParametrizedLine<double, 2> project3DLine(const Eigen::Affine3d & affine,
                                                 const Eigen::ParametrizedLine<double, 3>& line);




#endif /* defined(__PointLineReloc__vpgl_algo__) */
