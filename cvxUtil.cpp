//
//  cvxUtil.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-27.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvxUtil.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>


vector<double>
CvxUtil:: generateRandomNumbers(double min_val, double max_val, int rnd_num)
{
    assert(rnd_num > 0);
    
    cv::RNG rng;
    vector<double> data;
    for (int i = 0; i<rnd_num; i++) {
        data.push_back(rng.uniform(min_val, max_val));
    }
    return data;
}
