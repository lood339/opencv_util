//
//  cvx_tensor.h
//  Relocalization
//
//  Created by jimmy on 2017-08-24.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef Relocalization_cvx_tensor_h
#define Relocalization_cvx_tensor_h

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <unsupported/Eigen/CXX11/Tensor>

using cv::Mat;

namespace cvx {    
    void cv2eigen(const Mat& src, Eigen::Tensor<unsigned char, 3>& dst);
    
} // name space

#endif
