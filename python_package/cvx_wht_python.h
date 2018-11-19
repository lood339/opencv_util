//
//  cvx_wht_python.h
//  WHT_wrapper
//
//  Created by jimmy on 11/18/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef __WHT_wrapper__cvx_wht_python__
#define __WHT_wrapper__cvx_wht_python__

#include <iostream>


extern "C" {
    
    /*
     @brief extrat WHT feature from a color image.
     im_data:
     rows:
     cols:
     channels:
     points:  N * 2, feautre location
     point_num: N
     patch_size: 4, 8, 16, 32, 64, 128
     kernel_num: redused dimension 32, 64
     features: N * (kernel_num * 3 - 3)
     */    
    void extractWHTFeature(unsigned char* im_data,
                        const int rows,
                        const int cols,
                        const int channles,
                        const double* points,
                        const int point_num,
                        const int patch_size,
                        const int kernel_num,
                        double* features); // output
    
}

#endif /* defined(__WHT_wrapper__cvx_wht_python__) */
