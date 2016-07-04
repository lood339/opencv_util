//
//  cvxEntropy.hpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvxEntropy_cpp
#define cvxEntropy_cpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml.hpp>
#include <vector>

using std::vector;
// calculate entropy of different models
class CvxEntropy
{
public:
    
    // cov: covariance matrix from one guassian
    static double computeGaussianEntropy(const cv::Mat & cov);
    
    // em_model: gaussian mixture model
    static double conputeGMMEntropyFirstOrderAppro(const cv::ml::EM & em_model);
    
    // index of minimum entropy
    static int argminEntropy(const vector<cv::Mat> & covs);
    
    // duplicate the row of mat
    static void duplicateMatrix(cv::Mat & inoutmat);
    
    
};

#endif /* cvxEntropy_cpp */
