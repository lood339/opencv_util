//
//  cv_draw.hpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2016-01-11.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cv_draw_cpp
#define cv_draw_cpp

// util functions for draw images
#include <stdio.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"

using namespace::std;

class CvDraw
{
public:
    
    static void draw_match_vertical(const cv::Mat &image1,
                                    const cv::Mat &image2,
                                    const vector< cv::Point2d > & pts1,
                                    const vector< cv::Point2d > & pts2,
                                    cv::Mat & matches,
                                    const int sample_num = 1);    
};



#endif /* cv_draw_cpp */
