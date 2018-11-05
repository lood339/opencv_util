//
//  eigen_sg_smooth.h
//  Classifer_RF
//
//  Created by jimmy on 2017-03-07.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__eigen_sg_smooth__
#define __Classifer_RF__eigen_sg_smooth__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using namespace std;
class EigenSGSmooth
{
public:
    // window_size: 16
    // order: 1
    static void smooth(const double* data_in, int data_length, vector<double> & data_out, int window_size, int order);
    
    // window_size: 16
    // order: 1
    static void smooth(vector<double> & data_in_out, int window_size, int order);
    
    // smooth each column of the data
    static void smoothEachColumn(Eigen::MatrixXd & data_in_out, int window_size, int order);
    
};

#endif /* defined(__Classifer_RF__eigen_sg_smooth__) */
