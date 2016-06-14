//
//  cvxCalib3d.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-12.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxCalib3d.hpp"
#include "Kabsch.h"

void CvxCalib3D::rigidTransform(const vector<cv::Point3d> & src, const cv::Mat & affine, vector<cv::Point3d> & dst)
{
    cv::Mat rot   = affine(cv::Rect(0, 0, 3, 3));
    cv::Mat trans = affine(cv::Rect(3, 0, 1, 3));
    
    for (int i = 0; i<src.size(); i++) {
        cv::Mat p = rot * cv::Mat(src[i]) + trans;
        dst.push_back(cv::Point3d(p));
    }    
}

void CvxCalib3D::KabschTransform(const vector<cv::Point3d> & src, const vector<cv::Point3d> & dst, cv::Mat & affine)
{
    assert(src.size() == dst.size());
    assert(src.size() >= 4);
    
    Eigen::Matrix3Xd in(3, src.size());
    Eigen::Matrix3Xd out(3, dst.size());
    
    for (int i = 0; i<src.size(); i++) {
        in(0, i) = src[i].x;
        in(1, i) = src[i].y;
        in(2, i) = src[i].z;
        out(0, i) = dst[i].x;
        out(1, i) = dst[i].y;
        out(2, i) = dst[i].z;
    }
    Eigen::Affine3d aff = Find3DAffineTransform(in, out);
    affine = cv::Mat::zeros(3, 4, CV_64FC1);
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<4; j++) {
            affine.at<double>(i, j) = aff(i, j);
        }
    }
}