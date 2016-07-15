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
    
    double band_width = param.band_width_;
    int min_size = param.min_size;
    double min_dis = param.min_dis;
    assert(rows/min_size < 100);
    
    CMeanShift<double> ms;
    ms.setBandwidth(band_width);
    
    CEuclideanGeometry<double> geom(cols);
    CVectorPointSet<double> dataPoints(cols, rows, (double *)data.ptr());
    CVectorPointSet<double> unprunedModes(cols, rows);
    
    ms.doMeanShift(geom, dataPoints, unprunedModes);
    
 //   cout<<"un pruned mode number is "<<unprunedModes.size()<<endl;
    
    CVectorPointSet<double> prunedModes(cols, 100);
    vector<int> mode_sizes;
    ms.pruneModes(geom, unprunedModes, prunedModes, mode_sizes, min_size, min_dis);
    assert(mode_sizes.size() == prunedModes.size());
    
  //  cout<<"pruned mode number is "<<prunedModes.size()<<endl;
    for (int i = 0; i<prunedModes.size(); i++) {
        cv::Mat m(1, cols, CV_64FC1);
        for (int j = 0; j<cols; j++) {
            m.at<double>(0, j) = prunedModes[i][j];
        }
        modes.push_back(m);
        wt.push_back(1.0 * mode_sizes[i]/rows);
    }
    //double kernelDensities[100];
    //ms.getKernelDensities(geom, dataPoints, prunedModes, kernelDensities);    
    
    
    return 0;
}
