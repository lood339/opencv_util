//
//  cvx_pose_estimation_uncertainty.h
//  PointLineReloc
//
//  Created by jimmy on 2017-05-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_pose_estimation_uncertainty__
#define __PointLineReloc__cvx_pose_estimation_uncertainty__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

using std::vector;


class CvxPoseEstimationUncertainty
{
public:
    struct PreemptiveRANSACUncParameter
    {
        double dist_threshold_;    // distance threshod, unit meter
        int sample_num_;
        
    public:
        PreemptiveRANSACUncParameter()
        {
            dist_threshold_ = 0.1;  // meter
            sample_num_ = 500;
        }        
    };
    
public:
    // world_pts_covariance: covariance matrix, make sure they are invertable
    static bool preemptiveRANSAC3DOneToMany(const vector<Eigen::Vector3d>& camera_pts,
                                            const vector<vector<Eigen::Vector3d> >& world_pts,
                                            const vector<vector<Eigen::Matrix3d> >& world_pts_covariance,
                                            const PreemptiveRANSACUncParameter & param,
                                            Eigen::Affine3d& camera_pose);
};

#endif /* defined(__PointLineReloc__cvx_pose_estimation_uncertainty__) */
