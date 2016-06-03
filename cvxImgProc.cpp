//
//  cvxImgProc.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-06-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxImgProc.hpp"

Mat CvxImgProc::gradientOrientation(const Mat & img)
{
    assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);
    
    Mat src_gray;
    Mat grad;
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
    /// Gradient X
    Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
    /// Gradient Y
    Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
    
    grad_x.convertTo(grad_x, CV_32FC1);
    grad_y.convertTo(grad_y, CV_32FC1);
    
    Mat orientation;
    cv::phase(grad_x, grad_y, orientation, false);
    return orientation;
}