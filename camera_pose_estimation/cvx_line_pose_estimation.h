//
//  cvx_line_pose_estimation.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-23.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_line_pose_estimation__
#define __PointLineReloc__cvx_line_pose_estimation__

// camera pose estimation from line correspondences
#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

using std::vector;
using Eigen::Vector3d;

class CvxLinePoseEstimation
{
public:
    struct PreemptiveRANSACLine3DParameter
    {
        double point_dist_threshold;              // distance threshod, unit meter
        double point_to_line_dist_threshold;
        
    public:
        PreemptiveRANSACLine3DParameter()
        {
            point_dist_threshold = 0.1;
            point_to_line_dist_threshold = 0.1;
        }
    };
public:
    // world_pts: world coordinate locations in each lines, has noise
    // world_line_segments: predicted value, has noise
    // camera_line_segments: lines observed in camera coordinate, observed value, has noise but smaller than world_line_segments
    // camera_pose: 3 x 4 camera pose, from world to camera
    static bool preemptiveRANSAC3DOneToMany(const vector<vector<Eigen::Vector3d> > & world_line_pts,
                                            const vector<vector<Eigen::Vector3d> > & camera_line_pts,
                                            const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & world_line_segments,
                                            const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & camera_line_segments,
                                            const PreemptiveRANSACLine3DParameter & param,
                                            Eigen::Affine3d& camera_pose);
    
};

#endif /* defined(__PointLineReloc__cvx_line_pose_estimation__) */
