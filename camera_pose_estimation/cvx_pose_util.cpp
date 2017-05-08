//
//  cvx_pose_util.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-22.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_pose_util.h"
#include <iostream>

using std::cout;
using std::endl;

void poseDistance(const Eigen::Affine3d & src_pose, const Eigen::Affine3d & dest_pose,
                  double & angle_distance,
                  double & space_disance)
{
    double scale = 180.0/3.14159;
    
    Eigen::Matrix3d rot1 = src_pose.rotation();
    Eigen::Matrix3d rot2 = dest_pose.rotation();
    
    Eigen::Quaternion<double> q1(rot1);
    Eigen::Quaternion<double> q2(rot2);
    
    double val_dot = fabs(q1.dot(q2));
    angle_distance = 2.0 * acos(val_dot) * scale;
    
    space_disance = (src_pose.translation() - dest_pose.translation()).norm();
}

double angleDistance(const Eigen::Matrix3d & src_rot, const Eigen::Matrix3d& dst_rot)
{
    double scale = 180.0/3.14159;
    
    Eigen::Quaternion<double> q1(src_rot);
    Eigen::Quaternion<double> q2(dst_rot);
    
    double val_dot = fabs(q1.dot(q2));
    val_dot = std::min(1.0, val_dot);
    val_dot = std::max(-1.0, val_dot);
    return 2.0 * acos(val_dot) * scale;    
}
