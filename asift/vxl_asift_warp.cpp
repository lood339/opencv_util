//
//  vxl_asift_warp.cpp
//  MAVGoogleImageMatching
//
//  Created by jimmy on 2015-10-30.
//  Copyright © 2015 jimmy. All rights reserved.
//

#include "vxl_asift_warp.hpp"
#include "cvx_asift_warp.hpp"
#include "vxlOpenCV.h"

vnl_matrix_fixed<double, 2, 3> vxl_asift_warp::warp_image(const vil_image_view<vxl_byte> & image,
                                                          double rotate,
                                                          double tilt,
                                                          vil_image_view<vxl_byte> & warped_image)
{
    Mat cvImage = VxlOpenCVImage::cv_image(image);
    Mat cvWarpedImage;
    Mat affine = cvx_asift_warp::warp_image(cvImage, rotate, tilt, cvWarpedImage);
    
    warped_image = VxlOpenCVImage::to_vil_image_view(cvWarpedImage);
    assert(affine.rows == 2 && affine.cols == 3);
    
    vnl_matrix_fixed<double, 2, 3> affine_vxl;
    for (int i = 0; i<2; i++) {
        for (int j = 0; j<3; j++) {
            affine_vxl(i, j) = affine.at<double>(i, j);
        }
    }
    return affine_vxl;
}