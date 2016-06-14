//
//  cvxCalib3d.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-12.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxCalib3d_cpp
#define cvxCalib3d_cpp

#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using std::vector;

class CvxCalib3D
{
public:    
    static void rigidTransform(const vector<cv::Point3d> & src, const cv::Mat & affine, vector<cv::Point3d> & dst);
    static void KabschTransform(const vector<cv::Point3d> & src, const vector<cv::Point3d> & dst, cv::Mat & affine);
};

#endif /* cvxCalib3d_cpp */
