//
//  cvxImgProc.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxImgProc.hpp"
#include "cv_draw.hpp"

Mat CvxImgProc::gradientOrientation(const Mat & img, const int gradMagThreshold)
{
    assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);
    
    Mat src_gray;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    GaussianBlur( img, img, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    /// Convert it to gray
    if (img.channels() == 3) {
        cvtColor( img, src_gray, CV_BGR2GRAY );
    }
    else {
        src_gray = img;
    }
    
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;
    /// Gradient X
    Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    grad_x.convertTo(grad_x, CV_64FC1);
    grad_y.convertTo(grad_y, CV_64FC1);
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    cv::threshold(grad, grad, gradMagThreshold, 1.0, cv::THRESH_BINARY); // grad 0 or 1
    grad.convertTo(grad, CV_64FC1);
    
    Mat orientation;
    cv::phase(grad_x, grad_y, orientation, false);
    
    Mat threshold_orientation = orientation.mul(grad); // set small gradient positions as zero, other graduent as the same
    
 //   cv::Mat vmat = CvDraw::visualize_gradient(grad, threshold_orientation);
 //   cv::imshow("orientation", vmat);
 //   cv::waitKey();
    
    return threshold_orientation;
}