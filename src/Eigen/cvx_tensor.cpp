//
//  cvx_tensor.cpp
//  Relocalization
//
//  Created by jimmy on 2017-08-24.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_tensor.h"

namespace cvx {
    void cv2eigen(const Mat& src, Eigen::Tensor<unsigned char, 3>& dst)
    {
        assert(src.channels() == 3);
        assert(src.type() == CV_8UC3);
        
        int h = src.rows;
        int w = src.cols;
        
        dst = Eigen::Tensor<unsigned char, 3>(h, w, 3);
        
        // copy
        for (auto y = 0; y<h; y++) {
            for (auto x = 0; x<w; x++) {
                dst(y, x, 0) = src.at<cv::Vec3b>(y, x)[0];
                dst(y, x, 1) = src.at<cv::Vec3b>(y, x)[1];
                dst(y, x, 2) = src.at<cv::Vec3b>(y, x)[2];
            }
        }
    }

    
} // name space