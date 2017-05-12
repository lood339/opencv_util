//
//  cvxNormals.h
//  RGBD_RF
//
//  Created by jimmy on 2016-11-25.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__cvxNormals__
#define __RGBD_RF__cvxNormals__

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"

// calculate normal direction from rgbd images
class CvxNormals
{
public:
    // camera_matrix, depth, CV_64F
    static void compute_normals(const cv::Mat & camera_matrix, const cv::Mat & depth, cv::Mat & normal);
    
};


#endif /* defined(__RGBD_RF__cvxNormals__) */
