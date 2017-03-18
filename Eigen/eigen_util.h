//
//  eigen_util.h
//  Classifer_RF
//
//  Created by jimmy on 2017-03-13.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__eigen_util__
#define __Classifer_RF__eigen_util__

#include <stdio.h>
#include <vector>

using std::vector;

class EigenUtil
{
public:
    // data is in a dontinuous
    // smooth data using L2 norm, default (w1, w2, w3) = (1.0, 1.0, 100.0)
    static bool L2NormSmooth(const vector<double> & data, vector<double> & smoothed_data, double w1, double w2, double w3);
    
    // data in a discontinuous time space
    static bool L2NormSommth(const vector<double> & data, const vector<int> time_stamp,
                             vector<double> & smoothed_data, const double w1, const double w2, const double w3);
};




#endif /* defined(__Classifer_RF__eigen_util__) */
