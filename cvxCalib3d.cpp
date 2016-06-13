//
//  cvxCalib3d.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-12.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxCalib3d.hpp"

void CvxCalib3D::rigidTransform(const vector<cv::Point3d> & src, const cv::Mat & affine, vector<cv::Point3d> & dst)
{
    cv::Mat rot   = affine(cv::Rect(0, 0, 3, 3));
    cv::Mat trans = affine(cv::Rect(3, 0, 1, 3));
    
    for (int i = 0; i<src.size(); i++) {
        cv::Mat p = rot * cv::Mat(src[i]) + trans;
        dst.push_back(cv::Point3d(p));
    }    
}