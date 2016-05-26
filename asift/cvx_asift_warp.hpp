//
//  cvx_asift_warp.hpp
//  ASIFT
//
//  Created by jimmy on 2015-10-21.
//  Copyright © 2015 jimmy. All rights reserved.
//

#ifndef cvx_asift_warp_cpp
#define cvx_asift_warp_cpp

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

// warp image as in ASIFT: A New Framework for Fully Affine Invariant Image Comparison
using std::vector;

class cvx_asift_warp
{    
public:
    
    // rotate: [-180, 180] degree
    // tilt  : [0, 90], along y = h/2 axis degree 45, 60
    // warped_image: result
    // return: affine matrix applited to the image (not done yet)
    static cv::Mat warp_image(const cv::Mat & image, double rotate, double tilt, cv::Mat & warped_image);
    
    // apply affine transform to 2d point array
    static vector<cv::Vec2d> affine_warp_points(const vector<cv::Vec2d> & points, const cv::Mat & affine_transform);
    
    // draw lines between matched points
    static cv::Mat draw_match(const cv::Mat & image1, const cv::Mat & image2,
                              const vector<cv::Vec2d> & points1,
                              const vector<cv::Vec2d> & points2);
    
private:
    
    // tilt along the axis y = h/2
    // tilt [0, 90] degree
    // warped_image: output, different size
    // return: affine matrix applited to the image
    static cv::Mat compute_tilt_image(const cv::Mat & image, const double tilt, cv::Mat & warped_image);
    // rotate around the image center
    // rotation:[-180, 180] degree, positive --> anti-clock wise
    // warped_image: output different size
    // return: affine matrix applited to the image
    static cv::Mat compute_rotation_image(const cv::Mat & image, const double rotation, cv::Mat & warped_image);
};

#endif /* cvx_asift_warp_cpp */
