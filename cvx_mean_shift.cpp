//
//  cvx_mean_shift.cpp
//  PointLineReloc
//
//  Created by jimmy on 2017-04-14.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "cvx_mean_shift.h"
#include "ms.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <unordered_map>

using std::unordered_map;
// is the same mode in the tolerance?
static bool isSimilarMode(const Eigen::VectorXf& mode1, double* mode2, int dim, double tolerance)
{
    for(int i = 0; i < dim; i++) {
        if(fabs(mode1[i] - mode2[i]) > tolerance){
            return false;
        }
    }
    return true;
}

CvxMeanShift::Mode CvxMeanShift::createMode(const Eigen::VectorXf& m, const vector<Eigen::VectorXf>& input_data,
                                            const vector<unsigned int>& support_index)
{
    const int n = (int)support_index.size();
    const int dim = (int)input_data.front().size();
    
    //calculate covariance matrix of data points with pre-calculated mean
    cv::Mat_<float> covar_data(n, dim);
    cv::Mat_<float> covar_mean(1, dim);
    for (int d = 0; d < dim; d++) {
        covar_mean(0, d) = m[d];
    }
    
    cv::Mat_<float> covar_matrix;
    for(unsigned i = 0; i < support_index.size(); i++){
        unsigned int index = support_index[i];
        assert(index >= 0 && index < input_data.size());
        for(unsigned j = 0; j < dim; j++) {
            covar_data(i, j) = (float) input_data[index][j];
        }
    }
    
    cv::calcCovarMatrix(covar_data, covar_matrix, covar_mean, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE | CV_COVAR_USE_AVG, CV_32F);
    
    CvxMeanShift::Mode mode;
    mode.support = (unsigned int)support_index.size();
    mode.mean  = Eigen::VectorXf(dim);
    mode.covar = Eigen::MatrixXf(dim, dim);
    for (int i = 0; i<dim; i++) {
        mode.mean[i] = covar_mean(0, i);
        for (int j = 0; j<dim; j++) {
            mode.covar(i, j) = covar_matrix(i, j);
        }
    }
    return mode;
}


// code is modifed from "Uncertainty-Driven 6D Pose Estimation of Objects and Scenes from a Sin- gle RGB Image", cvpr 2016
// http://cvlab-dresden.de/research/scene-understanding/pose-estimation/
bool CvxMeanShift::meanShift(const vector<Eigen::VectorXf>& input_data,
                             const MeanShiftParameter& param,
                             vector<Mode> & modes)
{
    assert(input_data.size() > 0);
    const int dim = (int)input_data.front().size();
    const int n   = (int)input_data.size();
    const float band_width = (float)param.band_width;
    const double tolerance = param.min_dis;
    int min_support = param.min_size;
    
    MeanShift ms;
    
    kernelType kT[] = {Gaussian}; // kernel type
    float h[] = {band_width}; // bandwidth in millimeters
    int P[] = {dim};          // subspace definition, we have only 1 space of dimension dim
    int kp = 1;               // subspace definition, we have only 1 space of dimension dim
    ms.DefineKernel(kT, h, P, kp);
    
    // input data
    float* data = new float[n*dim];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < dim; j++) {
            data[i * dim + j] = input_data[i][j];
        }
    }
    ms.DefineInput(data, n, dim);
    
    // find modes
    std::vector<Eigen::VectorXf> temp_modes;        // initial list of modes (including very small ones)
    std::vector<std::vector<unsigned int> > support_index;   // list of support
    
    double* mode = new double[dim];  // mode we want to find
    double* point = new double[dim];  // point where we start the search (= sample point)
    
    for(int i = 0; i < n; i++)  {
        for(int j = 0; j < dim; j++) {
            mode[j] = 0.f;
            point[j] = input_data[i][j];
        }
        
        ms.FindMode(mode, point); // gets the mode center the sample wanders of to
        
        // assign the mode according to the mode center found or create a new one
        bool mode_found = false;
        
        for(unsigned j = 0; j < temp_modes.size(); j++) // iterate over all modes previously found
        {
            if(isSimilarMode(temp_modes[j], mode, dim, tolerance))
            {
                // found current mode in the list, add current sample to its support
                support_index[j].push_back(i);
                mode_found = true;
                break;
            }
        }
        
        if(!mode_found)
        {
            // mode not encountered before, create a new one with current sample as support
            Eigen::VectorXf new_mode = Eigen::VectorXf(dim);
            for(int j = 0; j < dim; j++){
                new_mode[j] = mode[j];
            }
            
            temp_modes.push_back(new_mode);
            
            //std::vector<int> new_support_index;
            //new_support_index.push_back(i);
            support_index.push_back(std::vector<unsigned int>(1, i));
        }
    }
    
    // clean up
    delete [] mode;
    mode = NULL;
    delete [] point;
    point = NULL;
    delete [] data;
    data = NULL;
    
    float support_min_ratio = 0.5f;        // relative mode support size (relative to the largest support) under which modes are discarded
    // determine mode with largest support
    int largest_support_num = 0;
    for(int s = 0; s < support_index.size(); s++){
        if(support_index[s].size() > largest_support_num) {
            largest_support_num = (int)support_index[s].size();
        }
    }
    if (largest_support_num < min_support) {
        //printf("largest support number is %d, threshold number is %d \n", largest_support_num, min_support);
        return false;
    }
    
    // calculate effective support minimum frow absolute and relative threshold
    min_support = std::max(min_support, (int)(support_min_ratio * largest_support_num));
    
    for(int s = 0; s < support_index.size(); s++) {
        // fit a GMM component and store this mode
        if(support_index[s].size() >= min_support){
            modes.push_back(CvxMeanShift::createMode(temp_modes[s], input_data, support_index[s]));
            modes.back().setSupportIndex(support_index[s]);
        }
    }
    
    return true;
}

bool CvxMeanShift::meanShift(const vector<Eigen::VectorXf>& input_data,
                             const MeanShiftParameter& param,
                             const vector<unsigned int>& data_index,
                             const int max_num_of_data,
                             vector<CvxMeanShift::Mode> & modes)
{
    assert(data_index.size() <= input_data.size());
    
    vector<int> sampled_index;   // data index
    for (int i = 0; i<data_index.size(); i++) {
        sampled_index.push_back(data_index[i]);
    }
    
    // randomly sample part of data
    if (sampled_index.size() > max_num_of_data) {
        std::random_shuffle(sampled_index.begin(), sampled_index.end());
        sampled_index.resize(max_num_of_data);
    }
    
    vector<Eigen::VectorXf> ms_data;  // mean shift data
    for (unsigned int i = 0; i<sampled_index.size(); i++) {
        ms_data.push_back(input_data[sampled_index[i]]);
    }
    CvxMeanShift::meanShift(ms_data, param, modes);
    
    // re-index sample index
    for (int i = 0; i<modes.size(); i++) {
        for (int j = 0; j<modes[i].support_indices.size(); j++) {
            int index = modes[i].support_indices[j];
            int re_index = sampled_index[index];  // this is index of input_data
            assert(re_index >= 0 && re_index < input_data.size());
            modes[i].support_indices[j] = re_index;
        }
    }
    
    return true;
}