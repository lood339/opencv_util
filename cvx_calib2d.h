//
//  cvx_calib2d.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-05-23.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__cvx_calib2d__
#define __CalibMeMatching__cvx_calib2d__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <vector>

using std::vector;
using cv::Mat;

namespace cvx {
    // find homography from a sequence of frame correspondences using ransac
    vector<Mat> findHomography(const vector<vector<cv::Point2f> >& src_points,
                               const vector<vector<cv::Point2f> >& dst_points,
                               vector<unsigned char>& mask,
                               int method = 0,
                               double ransac_reproj_threshold = 3.0,
                               const int max_iters = 2000,
                               const double confidence = 0.995);
    
    // default image size 1280 x 720
    // find inter frame homography using both point and lines
    // method: 0 --> point, 1 --> point and edge
    // output_warp: estimated homography,
    // warp_quality: space coverate of inlier edge center point, input and output, negative value for ignore this value
    // search length: parameter in point-on-line correspondence, pixel number on side of line
    // block_size: size of block used in computing the distance in line segment tracking
    // assumption: source image and destination image has narrow baseline (from similar view)
    bool findHomography(const cv::Mat& src_image,
                        const cv::Mat& dst_image,
                        const int method,                       
                        cv::Mat & output_warp,
                        double& warp_quality,
                        double search_length = 10,
                        int block_size = 25);
    
    // default image size 1280 x 720
    // find inter frame homography using both point and lines
    // src_points:
    // dst_points: points correspondence frome source and destination image
    // method: 0 --> point, 1 --> point and edge
    // output_warp: estimated homography,
    // warp_quality: space coverate of inlier edge center point, input and output, negative value for ignore this value
    // search length: parameter in point-on-line correspondence, pixel number on side of line
    // block_size: size of block used in computing the distance in line segment tracking
    // assumption: source image and destination image has Wide baseline (from different view)
    bool findHomography(const cv::Mat& src_image,
                        const cv::Mat& dst_image,
                        const vector<cv::Point2f>& src_points,
                        const vector<cv::Point2f>& dst_points,
                        const int method,
                        cv::Mat & output_warp,
                        double& warp_quality,
                        double search_length = 10,
                        int block_size = 25);    
    
}


#endif /* defined(__CalibMeMatching__cvx_calib2d__) */
