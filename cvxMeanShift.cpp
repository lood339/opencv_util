//
//  cvxMeanShift.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-07-15.
//  Copyright (c) 2016 jimmy. All rights reserved.
//

#include "cvxMeanShift.h"
#include "MeanShift.h"
#include "EuclideanGeometry.h"
#include "VectorPointSet.h"
#include <iostream>

using std::cout;
using std::endl;

int CvxMeanShift::meanShift(const cv::Mat & data,
                            vector<cv::Mat> & modes,
                            vector<double> & wt,
                            const CvxMeanShiftParameter & param)
{
    assert(data.type() == CV_64FC1);
    assert(data.isContinuous());
    int rows = data.rows;
    int cols = data.cols;
    
    // get mean shift parameters
    double band_width = param.band_width_;
    int min_size = param.min_size_;
    double min_dis = param.min_dis_;
    const int max_mode_num = param.max_mode_num_;
    
    CMeanShift<double> ms;
    ms.setBandwidth(band_width);
    
    CEuclideanGeometry<double> geom(cols);
    CVectorPointSet<double> dataPoints(cols, rows, (double *)data.ptr());
    CVectorPointSet<double> unprunedModes(cols, rows);
    
    ms.doMeanShift(geom, dataPoints, unprunedModes);
    
    int dim = cols;
    //cout<<"un pruned mode number is "<<unprunedModes.size()<<endl;
    
    CVectorPointSet<double> prunedModes(cols, max_mode_num);
    vector<int> mode_sizes;
    ms.pruneModes(geom, unprunedModes, prunedModes, max_mode_num, mode_sizes, min_size, min_dis);
    assert(mode_sizes.size() == prunedModes.size());
    
    cout<<"pruned mode number is "<<prunedModes.size()<<endl;
    for (int i = 0; i<prunedModes.size() && i < mode_sizes.size(); i++) {
        cv::Mat m(dim, 1, CV_64FC1);
        for (int j = 0; j<dim; j++) {
            m.at<double>(j, 0) = prunedModes[i][j];
        }
        modes.push_back(m);
        wt.push_back(1.0 * mode_sizes[i]/rows);
    }    
    //double kernelDensities[100];
    //ms.getKernelDensities(geom, dataPoints, prunedModes, kernelDensities);
    return (int)modes.size();
}

bool
CvxMeanShift::meanShiftLargestMode(const cv::Mat & data,
                                   cv::Mat & mode,
                                   double & wt,
                                   const CvxMeanShiftParameter & param)
{
    vector<cv::Mat> modes;
    vector<double> wts;
    int num = CvxMeanShift::meanShift(data, modes, wts, param);
    if (num == 0) {
        return false;
    }
    int idx_max = (int)std::distance(wts.begin(), std::max_element(wts.begin(), wts.end()));
    assert(idx_max >= 0 && idx_max < wts.size());
    mode = modes[idx_max];
    wt  = wts[idx_max];
    return true;
}
