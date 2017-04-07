//
//  cvx_pl_pose_estimation.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-06.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_pl_pose_estimation__
#define __PointLineReloc__cvx_pl_pose_estimation__

// camera pose estimation from point and line locations
#include <stdio.h>
#include <vector>
#include "cvxImage_310.hpp"
#include <Eigen/Dense>


using std::vector;

struct PreemptiveRANSAC3DPointLineParameter
{
    double dis_threshold_;              // distance threshod, unit meter
    double line_distance_threshold;     // distance between end points to line, unit meter
    double line_loss_weight;                 // one line vs one point
    
public:
    PreemptiveRANSAC3DPointLineParameter()
    {
        dis_threshold_ = 0.1;
        line_distance_threshold = 0.05;
        line_loss_weight = 4.0;
    }
    
};


class CvxPLPoseEstimation
{
public:
    
    // camera_pts: camera coordinate locations
    // candidate_wld_pts: corresonding world coordinate locations, estimated points, had outliers, multiple choices
    // line_start_points, line_end_points: line end points in camera coordinate
    // candidate_wld_lines: predicted world line
    // camera_pose: 4 x 4 camera pose, from camera to world
    static bool preemptiveRANSAC3DOneToMany(const vector<cv::Point3d> & camera_pts,
                                            const vector<vector<cv::Point3d> > & candidate_wld_pts,
                                            const vector<cv::Point3d> & line_start_points,
                                            const vector<cv::Point3d> & line_end_points,
                                            const vector<Eigen::ParametrizedLine<double, 3> > & candidate_wld_lines,
                                            const PreemptiveRANSAC3DPointLineParameter & param,
                                            cv::Mat & camera_pose);
    
    
};

#endif /* defined(__PointLineReloc__cvx_pl_pose_estimation__) */
