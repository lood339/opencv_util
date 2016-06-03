//
//  cvxImgProc.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxImgProc_cpp
#define cvxImgProc_cpp

#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"

using std::vector;
using cv::Mat;

class CvxImgProc
{
public:
    // gradient orientation in [0, 2 * pi)
    static Mat gradientOrientation(const Mat & img);
};

#endif /* cvxImgProc_cpp */
