//
//  cvx_pose_util.h
//  PointLineReloc
//
//  Created by jimmy on 2017-04-22.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PointLineReloc__cvx_pose_util__
#define __PointLineReloc__cvx_pose_util__

#include <stdio.h>
#include <Eigen/Dense>


// camera pose difference between two poses
// angle_distance: in degree
void poseDistance(const Eigen::Affine3d & src_pose, const Eigen::Affine3d & dest_pose,
                  double & angle_distance, double & space_disance);

double angleDistance(const Eigen::Matrix3d & src_rot, const Eigen::Matrix3d& dst_rot);

#endif /* defined(__PointLineReloc__cvx_pose_util__) */
