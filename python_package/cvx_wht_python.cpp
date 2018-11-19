//
//  cvx_wht_python.cpp
//  WHT_wrapper
//
//  Created by jimmy on 11/18/18.
//  Copyright (c) 2018 Nowhere Planet. All rights reserved.
//

#include "cvx_wht_python.h"

#include "cvx_walsh_hadamard.h"

extern "C" {
    void extractWHTFeature(unsigned char* im_data,
                        const int rows,
                        const int cols,
                        const int channles,
                        const double* points,
                        const int point_num,
                        const int patch_size,
                        const int kernel_num,
                        double* output_features)
    {
        assert(channles == 3);
        cv::Mat img(rows, cols, CV_8UC3, (void*)im_data);;
       
        
        /*
        static
        bool generateWHFeatureWithoutFirstPattern(const cv::Mat & rgb_image,
                                                  const vector<cv::Point2d> & pts,
                                                  const int patchSize,
                                                  const int kernelNum,
                                                  vector<Eigen::VectorXf> & features);
         */
        vector<cv::Point2d> pts;
        for (int i = 0; i<point_num; i++) {
            cv::Point2d p(points[2*i], points[2*i+1]);
            pts.push_back(p);
        }
        vector<Eigen::VectorXf> features;
        CvxWalshHadamard::generateWHFeatureWithoutFirstPattern(img,
                                                               pts,
                                                               patch_size, kernel_num, features);
        assert(features[0].size() == kernel_num * 3 - 3);
        // output
        const int dim = kernel_num * 3 - 3;
        for (int i = 0; i<features.size(); i++) {
            for (int j = 0; j<dim; j++) {
                output_features[i * dim + j] = features[i][j];
            }
        }
        
    }
    
}
