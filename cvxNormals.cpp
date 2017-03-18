//
//  cvxNormals.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-11-25.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "cvxNormals.h"
#include <opencv2/rgbd.hpp>

void CvxNormals::compute_normals(const cv::Mat & camera_matrix, const cv::Mat & depth, cv::Mat & normal)
{
    assert(camera_matrix.type() == CV_64FC1);
    assert(depth.type() == CV_64FC1);
    
    cv::rgbd::RgbdNormals::RgbdNormals normal_computer(depth.rows, depth.cols, depth.type(), camera_matrix, 5, cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
    normal_computer(depth, normal);
}