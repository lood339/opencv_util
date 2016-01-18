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
    /*
     input: image1, image2, RGB color image
            pts1, pts2, points in iamges
     output: matches, vertical image pairs of image1 and image2, pts1 and pts2 area connected by red lines
     sample_num: when the number of pts1 and pts2 are too large, sampling drawing points
     */
    static void draw_match_vertical(const cv::Mat &image1,
                                    const cv::Mat &image2,
                                    const vector< cv::Point2d > & pts1,
                                    const vector< cv::Point2d > & pts2,
                                    cv::Mat & matches,
                                    const int sample_num = 1);
    
    // draw cross around the point
    static void draw_cross(cv::Mat & image,
                           const vector<cv::Point2f> & pts,
                           const cv::Scalar & color,
                           const int length = 5);

    // error_image: output of parameter
    static void draw_reprojection_error(const vector<cv::Point3f> &pt_3d,
                                        const vector<cv::Point2f> &pt_2d,
                                        const cv::Mat & camera_intrinsic_matrix,
                                        const cv::Mat & rotation,
                                        const cv::Mat & translation,
                                        const cv::Mat & distCoeffs,
                                        vector<float> & reproj_errors,
                                        cv::Mat & error_image);
    
    static cv::Scalar red();
    static cv::Scalar green();
    static cv::Scalar blue();
};




#endif /* cv_draw_cpp */
