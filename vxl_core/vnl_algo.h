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
using Eigen::Vector2d;

class VnlAlgo
{
public:
    // camera_pts: unit meter
    // estimate caemra pose from point to point correspondence
    // and point on line correspondence
    // camera pose: camera coordinate to world coordinate [R:T]
    static
    bool estimateCameraPose(const vector<Vector3d> & camera_pts,
                            const vector<Vector3d> & world_pts,
                            const vector<Vector3d> & camera_line_start_pts,
                            const vector<Vector3d> & camera_line_end_pts,
                            const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,                            
                            const Eigen::Affine3d& init_camera,
                            Eigen::Affine3d& final_camera);
    
    // estimate camera pose from point on lines
    // camera pose: camera coordinate to world coordinate [R:T]
    // camera and world coordinate can be switched
    static
    bool estimateCameraPoseFromLine(const vector< vector<Vector3d> > & camera_line_pts,
                                    const vector<Eigen::ParametrizedLine<double, 3> > & world_lines,
                                    const Eigen::Affine3d& init_pose,
                                    Eigen::Affine3d& final_pose);
    
    
    // estimate camera pose from point to line
    // camera pose: camera coordinate to world coordinate [R:T]
    // camera and world coordinate can be switched
    static
    bool estimateCameraPoseUsingPointLine(const vector<Vector3d>& source_pts,
                                          const vector<Vector3d>& target_pts,
                                          const vector< vector<Vector3d> > & source_line_pts,
                                          const vector<Eigen::ParametrizedLine<double, 3> > & target_lines,
                                          const Eigen::Affine3d& init_pose,
                                          Eigen::Affine3d& final_pose);
    
    // refine camera pose by minimize Mahalanobis distance
    // source_pts: observed points location in camera coordinate
    // target_pts: predicted points in world coordiante, it has prediction errors,
    // which is described by covariance matrix
    static
    bool estimateCameraPoseWithUncertainty(const vector<Eigen::Vector3d>& source_pts, // camera points
                                           const vector<Eigen::Vector3d>& target_pts, // world coordiante points (predicted value)
                                           const vector<Eigen::Matrix3d>& target_pt_covariance_inv, // invert of world coordinate points covariance
                                           const Eigen::Affine3d& init_pose,
                                           Eigen::Affine3d& refined_pose);
    
    // refine camera pose by minimizing Mahalanobis distance using both points and lines
    // target_line_cov: line model covariance matrix
    // refined_pose: camera to world
    static
    bool estimateCameraPoseWithUncertainty(const vector<Eigen::Vector3d>& camera_pts, // camera points
                                           const vector<Eigen::Vector3d>& world_pts, // world coordiante points (predicted value)
                                           const vector<Eigen::Matrix3d>& world_pt_precision, // invert of world coordinate points covariance
                                           // point on the line
                                           const vector<vector< Eigen::Vector3d> > & world_line_pts_group,
                                           const vector<vector< Eigen::Matrix3d> >& world_line_pts_precision,
                                           const vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >& camera_lines,
                                                        // matrix size, 6 x 6
                                           const Eigen::Affine3d& init_pose,
                                           Eigen::Affine3d& refined_pose);  
    
                            
    
};


#endif /* defined(__PointLineReloc__vnl_algo__) */
