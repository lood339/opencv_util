//
//  cvx_opt_python.h
//  EstimatePTZTripod
//
//  Created by jimmy on 11/11/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#ifndef EstimatePTZTripod_cvx_opt_python_h
#define EstimatePTZTripod_cvx_opt_python_h


extern "C" {
    void estimateCommomCameraCenterAndRotation(const double* model_pts,
                                               const int rows,
                                               const int cols,
                                               const double* input_init_cameras,
                                               const int camera_num,
                                               const int camera_param_len,
                                               const double* input_init_common_rotation,
                                               double* opt_cameras,
                                               double* commom_center,
                                               double* commom_rotation);    
}

#endif
