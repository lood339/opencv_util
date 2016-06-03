//
//  cvxEntropy.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-26.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxEntropy.hpp"

using cv::Mat;

double CvxEntropy::computeGaussianEntropy(const cv::Mat & cov)
{
    assert(cov.rows == cov.cols);
    double det = cv::determinant(cov);
    assert(det > 0.0);
    double temp = pow(2.0 * M_PI * M_E, cov.rows);
    double entropy = 0.5 * log( temp * det);
    return entropy;
}

// paper: On Entropy Approximation for Gaussian Mixture Random Vectors 2008
// appendex A
double CvxEntropy::conputeGMMEntropyFirstOrderAppro(const cv::ml::EM & em_model)
{
    double entropy = 0.0;
    Mat estimated_means = em_model.getMeans();
    Mat estimated_weights = em_model.getWeights();
    
    assert(estimated_means.type() == CV_64F);
    assert(estimated_weights.type() == CV_64F);
    
    for(int i = 0; i<estimated_weights.cols; i++)
    {
        double log_likelihood = em_model.predict2(estimated_means.row(i), cv::noArray())[0];
        entropy += -estimated_weights.at<double>(0, i) * log_likelihood;
    }
    
    return entropy;
    
}