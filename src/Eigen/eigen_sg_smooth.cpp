//
//  eigen_sg_smooth.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-03-07.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//


#include "eigen_sg_smooth.h"
#include "sgsmooth.h"
#include <assert.h>

void EigenSGSmooth::smooth(const double* data_in, int data_length, vector<double> & data_out, int window_size, int order)
{
    assert(data_in);
    assert(data_length > window_size);
    
    data_out.resize(data_length);
    for (int i = 0; i<data_length; i++) {
        data_out[i] = data_in[i];
    }
    
    EigenSGSmooth::smooth(data_out, window_size, order);
}

void EigenSGSmooth::smooth(vector<double> & data, int window_size, int order)
{
    double *pSmoothed = calc_sgsmooth((int)data.size(), &data[0], window_size, order);
    assert(pSmoothed == &data[0]);
}

void EigenSGSmooth::smoothEachColumn(Eigen::MatrixXd & data_in_out, int window_size, int order)
{
    long rows = data_in_out.rows();
    long cols = data_in_out.cols();
    for (int c = 0; c < cols; c++) {
        vector<double> cur_data(rows);
        for (int r = 0; r<rows; r++) {
            cur_data[r] = data_in_out(r, c);
        }
        EigenSGSmooth::smooth(cur_data, window_size, order);
        for (int r = 0; r<rows; r++) {
            data_in_out(r, c) = cur_data[r];
        }
    }
}