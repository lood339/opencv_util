//
//  cvxMeanShift.h
//  RGB_RF
//
//  Created by jimmy on 2016-07-15.
//  Copyright (c) 2016 jimmy. All rights reserved.
//

#ifndef __RGB_RF__cvxMeanShift__
#define __RGB_RF__cvxMeanShift__

// wrap meanshift from "Nonlinear Mean Shift over Riemannian Manifolds " IJCV 2009

#include <stdio.h>
#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using std::vector;

struct CvxMeanShiftParameter
{
    double band_width_;
    int min_size;    // mininum size required for a mode
    double min_dis;  // The minimum distance required between modes
    
    CvxMeanShiftParameter()
    {
        band_width_ = 1.0;
        min_size = 50;
        min_dis = 0.1;
    }
    
};

class CvxMeanShift
{
public:
    // data: every row is a point
    // modes: output
    // wt: weight of each mode
    static int meanShift(const cv::Mat & data,
                         vector<cv::Mat> & modes,
                         vector<double> & wt,
                         const CvxMeanShiftParameter & param);
    
};

#endif /* defined(__RGB_RF__cvxMeanShift__) */
